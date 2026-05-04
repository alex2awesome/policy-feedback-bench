"""
Per-model comment-anticipation benchmark harness.

For a given LLM, runs the full pipeline:
  1) Generate claims for each of 558 benchmark dockets (vanilla OR +subgroup)
  2) Compute Axis A (provision targeting): Recall, Precision, Pearson on attention
  3) Compute Axis B (semantic alignment): pair LLM claims with real claims, label via gpt-5-mini judge
  4) Compute Axis C (frame distribution): label LLM claims with primary_frame
  5) Substantivity AUC: real-vs-LLM rank discrimination via Llama-3.1-8B classifier (sk3)
  6) Aggregate into one row of Table 2

Usage:
    OPENAI_API_KEY=$(cat ~/.openai-salt-lab-key.txt) \
        python scripts/run_benchmark_per_model.py \
            --model gpt-4o \
            --provider openai \
            --variant vanilla \
            --output-name gpt-4o_vanilla

    # subgroup-prompting variant
    OPENAI_API_KEY=$(cat ~/.openai-salt-lab-key.txt) \
        python scripts/run_benchmark_per_model.py --model gpt-4o --provider openai \
            --variant subgroup --output-name gpt-4o_subgroup

Outputs (under data/derived/benchmark_runs/<output_name>/):
    generated_claims.jsonl     # one row per docket, parsed_claims field
    pair_judgments.jsonl       # one row per (LLM-claim, real-claim) pair with label
    claim_frame_labels.jsonl   # one row per LLM claim with primary_frame
    metrics.json               # final aggregate row for the table

Note: substantivity AUC is computed by a separate sk3-side script (scripts/run_substantivity_for_run.py)
that ingests generated_claims.jsonl. Trigger it after the generation step.
"""
from __future__ import annotations
import argparse, asyncio, gzip, json, logging, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def _open_text(path):
    """open() that transparently handles .gz files."""
    p = Path(path)
    return gzip.open(p, 'rt') if p.suffix == '.gz' else open(p)

# Layout: this script lives at the package root; data lives in ./benchmark/.
HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get('POLICY_FEEDBACK_BENCH_ROOT', HERE))
BENCH = BASE / 'benchmark'
RUNS = BASE / 'reference_runs'

# Inputs (shipped in this release; produced by build_release.py from the source dataset)
PROPOSED_UNITS = BENCH / 'proposed_rule_provisions_558.jsonl'
EXISTING_BENCH = BENCH / 'benchmark_dockets_558.jsonl'  # 558-docket index
MATCH2_LABELED = BENCH / 'real_claims_matched_to_provisions_558.jsonl.gz'

# ----------------- prompts -----------------
GEN_PROMPT_VANILLA = """\
You are an expert policy analyst reviewing a proposed federal regulation. Below are the specific regulatory provisions (obligations, permissions, prohibitions, conditions, exceptions) extracted from the proposed rule.

Your task: Generate the public comments that stakeholders would likely submit during the comment period. For each comment, express it as a specific claim — a concrete point about a specific provision.

Format each claim on its own line, prefixed with the provision number it addresses:
[P1] claim about provision 1
[P3] claim about provision 3
[P1] another claim about provision 1

Generate as many distinct claims as you think real commenters would submit. Be specific — reference particular requirements, thresholds, or actors mentioned in the provisions. Consider perspectives from industry, advocacy groups, individuals, and legal experts.

PROPOSED RULE PROVISIONS:
{provisions}

Generate the anticipated public comments:"""

GEN_PROMPT_SUBGROUP = """\
You are an expert policy analyst. Below are regulatory provisions from a proposed federal rule. You will write public-comment claims FROM THE PERSPECTIVE OF A SPECIFIC STAKEHOLDER GROUP.

STAKEHOLDER PERSPECTIVE: {stakeholder_persona}

Your task: Generate the public-comment claims this stakeholder would submit. Each claim should be a concrete point about a specific provision, written in the voice and concerns characteristic of {stakeholder_label}.

Format each claim on its own line, prefixed with the provision number it addresses:
[P1] claim about provision 1
[P3] claim about provision 3

Be specific — reference particular requirements, thresholds, or actors mentioned in the provisions, AND write in a way that reflects this stakeholder's typical concerns, vocabulary, and framings.

PROPOSED RULE PROVISIONS:
{provisions}

{stakeholder_label} comments on this proposal:"""

STAKEHOLDER_PERSONAS = {
    'industry':   ('industry / corporation / trade association',
                   'a regulated company, trade association, or industry coalition. You focus on compliance cost, operational feasibility, market effects, and ROI. You cite economic data and industry-specific operational details.'),
    'labor':      ('labor union representative',
                   'a labor union representing workers in the regulated industry. You focus on worker safety, job protection, fair labor standards, and economic impact on workers. You cite worker statistics and on-the-job experience.'),
    'advocacy':   ('public-interest advocacy NGO',
                   'an environmental, civil rights, or public-interest advocacy organization. You focus on broader public good, environmental protection, equity, and the regulation\'s role in achieving social goals. You cite scientific studies, public-health data, and advocacy research.'),
    'expert':     ('expert / academic organization',
                   'an academic researcher, professional society, or expert organization. You focus on technical correctness, scientific accuracy, peer-reviewed evidence, and methodological soundness. You cite research, technical specifications, and disciplinary expertise.'),
    'gov':        ('government / state agency',
                   'a state or local government agency that interacts with this federal regulation. You focus on inter-jurisdictional coordination, implementation feasibility for state agencies, and federalism issues. You cite state-level data and operational details.'),
    'lawfirm':    ('law firm / bar association',
                   'a law firm or bar association representing regulated parties. You focus on statutory authority, constitutional issues, APA procedural requirements, due-process implications, and legal interpretation. You cite case law, statutes, and regulatory history.'),
    'individual': ('individual citizen or affected community member',
                   'an ordinary individual citizen affected by this rule. You focus on personal experience, lived impact on you or your family/community, and concrete harms or benefits you would face. You speak in first person about direct experience.'),
}

# ----------------- model adapters -----------------
async def call_openai(client, prompt, model, max_tokens=4000, reasoning_effort='low'):
    kwargs = dict(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        max_completion_tokens=max_tokens,
        extra_body={'service_tier': 'flex'} if 'gpt-5' in model else {},
    )
    if 'gpt-5' in model:
        kwargs['reasoning_effort'] = reasoning_effort
    resp = await client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or '').strip(), resp.usage.prompt_tokens, resp.usage.completion_tokens


async def call_anthropic(client, prompt, model, max_tokens=4000):
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = ''
    for block in resp.content:
        if hasattr(block, 'text'): text += block.text
    return text.strip(), resp.usage.input_tokens, resp.usage.output_tokens


def make_caller(provider: str, model: str, base_url: str | None = None):
    if provider in ('openai', 'vllm'):
        from openai import AsyncOpenAI
        if provider == 'vllm':
            # Local vLLM OpenAI-compatible server
            client = AsyncOpenAI(api_key='not-needed', base_url=base_url or 'http://localhost:8001/v1')
        else:
            client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        async def _call(prompt): return await call_openai(client, prompt, model)
        return _call
    elif provider == 'anthropic':
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        async def _call(prompt): return await call_anthropic(client, prompt, model)
        return _call
    raise ValueError(f'unknown provider: {provider}')


# ----------------- data loading -----------------
def load_benchmark_dockets():
    benchmark = []
    seen = set()
    for line in open(EXISTING_BENCH):
        r = json.loads(line)
        dk = r['docket_id']
        if dk in seen: continue
        seen.add(dk)
        benchmark.append({'docket_id': dk, 'n_units': r['n_units']})
    return benchmark


def load_units_per_docket():
    """For each benchmark docket, load the proposed-units in the same order as the existing benchmark prompt."""
    units = defaultdict(list)
    for line in open(PROPOSED_UNITS):
        r = json.loads(line)
        dk = r.get('docket_id')
        doc_id = r.get('doc_id', '')
        for u in r.get('units', []):
            units[dk].append({
                'global_id': f"{dk}::{doc_id}::{u['id']}",
                'section': u.get('section', '') or '',
                'type': u.get('type', '') or '',
                'summary': (u.get('action_summary') or '')[:300],
                'text': (u.get('obligation_text') or '')[:800],
            })
    return units


def format_provisions(units):
    parts = []
    for i, u in enumerate(units, 1):
        section = f"[{u['section']}]" if u['section'] else ''
        parts.append(f"P{i} {section} ({u['type']}): {u['summary'] or u['text'][:300]}")
    return '\n'.join(parts)


CLAIM_RE = re.compile(r'^\s*\[P(\d+)\]\s*(.+?)\s*$', re.MULTILINE)
def parse_claims(text, n_units):
    out = []
    for m in CLAIM_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= n_units:
            out.append({'provision_idx': idx, 'claim_text': m.group(2).strip()})
    return out


# ----------------- generation -----------------
async def _one_call_with_sem(call, sem, prompt, tag, docket_id):
    """A single API call with its own semaphore slot. Returns (raw, in_tok, out_tok, tag)."""
    async with sem:
        try:
            raw, ti, to = await call(prompt)
            return raw, ti, to, tag
        except Exception as e:
            logger.warning('call failed for %s/%s: %s', docket_id, tag, str(e)[:120])
            return '', 0, 0, tag


async def generate_for_docket(call, docket_id, units, variant, sem):
    """Returns list of generated claims with provision_idx + claim_text + (optional) stakeholder.

    Fires every API call (1 for vanilla, 7 for subgroup) concurrently and gathers them.
    Each call holds its own semaphore slot, so persona calls within a docket parallelize.
    """
    provisions = format_provisions(units)
    all_claims = []
    in_tok = out_tok = 0

    if variant == 'vanilla':
        raw, ti, to, _ = await _one_call_with_sem(
            call, sem,
            GEN_PROMPT_VANILLA.format(provisions=provisions),
            tag='vanilla', docket_id=docket_id)
        in_tok += ti; out_tok += to
        for c in parse_claims(raw, len(units)):
            c['stakeholder'] = None
            all_claims.append(c)
    else:  # subgroup
        # Build all 7 persona prompts and fire concurrently
        tasks = []
        for sg, (label, persona) in STAKEHOLDER_PERSONAS.items():
            prompt = GEN_PROMPT_SUBGROUP.format(
                provisions=provisions, stakeholder_persona=persona, stakeholder_label=label)
            tasks.append(_one_call_with_sem(call, sem, prompt, tag=sg, docket_id=docket_id))
        results = await asyncio.gather(*tasks)
        for raw, ti, to, sg in results:
            in_tok += ti; out_tok += to
            for c in parse_claims(raw, len(units)):
                c['stakeholder'] = sg
                all_claims.append(c)

    return all_claims, in_tok, out_tok


async def run_generation(model, provider, variant, output_dir, concurrency=20, base_url=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'generated_claims.jsonl'
    call = make_caller(provider, model, base_url=base_url)

    benchmark = load_benchmark_dockets()
    logger.info('Loaded %d benchmark dockets', len(benchmark))
    docket_units = load_units_per_docket()

    done_dks = set()
    if out_path.exists():
        for line in open(out_path):
            try: done_dks.add(json.loads(line)['docket_id'])
            except Exception: pass
    todo = [b for b in benchmark if b['docket_id'] not in done_dks]
    logger.info('Resume: %d already generated, %d remaining', len(done_dks), len(todo))

    sem = asyncio.Semaphore(concurrency)
    in_tok = out_tok = 0
    t0 = time.time()
    BATCH = 20
    with open(out_path, 'a') as fout:
        for s in range(0, len(todo), BATCH):
            batch = todo[s:s+BATCH]
            tasks = [generate_for_docket(call, b['docket_id'], docket_units[b['docket_id']],
                                          variant, sem) for b in batch]
            results = await asyncio.gather(*tasks)
            for b, (claims, ti, to) in zip(batch, results):
                rec = {'docket_id': b['docket_id'], 'n_units': b['n_units'],
                       'parsed_claims': claims, 'variant': variant, 'model': model}
                fout.write(json.dumps(rec) + '\n')
                in_tok += ti; out_tok += to
            fout.flush()
            elapsed = time.time() - t0
            n_done = s + len(batch)
            logger.info('%d/%d (%.1f/s) | tokens in=%d out=%d', n_done, len(todo),
                        n_done / max(elapsed, 1), in_tok, out_tok)
    logger.info('Done: %s | in=%d out=%d', out_path, in_tok, out_tok)


# ----------------- Axis A: provision targeting -----------------
def compute_axis_a(output_dir):
    gen_path = output_dir / 'generated_claims.jsonl'
    runs = [json.loads(l) for l in open(gen_path)]
    # Map docket -> provision_idx -> count of LLM claims
    docket_to_count = {}
    for r in runs:
        c = Counter()
        for cl in r.get('parsed_claims', []):
            idx = cl.get('provision_idx')
            if idx: c[idx] += 1
        docket_to_count[r['docket_id']] = c

    # Map docket -> provision_idx -> unit_id (using same ordering as the prompt)
    docket_units = load_units_per_docket()
    docket_to_idx2id = {dk: {i+1: u['global_id'] for i, u in enumerate(us)}
                       for dk, us in docket_units.items()}

    # Real provision attention from match2 YES
    benchmark_dks = set(docket_to_count.keys())
    docket_real = defaultdict(lambda: defaultdict(set))  # dk -> unit_id -> {cluster_uid}
    for line in _open_text(MATCH2_LABELED):
        if '"llm_label": 1' not in line: continue
        try: r = json.loads(line)
        except: continue
        if r.get('llm_label') != 1: continue
        dk = r.get('docket_id')
        if dk not in benchmark_dks: continue
        uid = r.get('unit_id'); cu = r.get('cluster_uid')
        if uid and cu:
            docket_real[dk][uid].add(cu)

    rows = []
    real_attn, llm_attn = [], []
    for dk in benchmark_dks:
        idx2id = docket_to_idx2id.get(dk, {})
        llm_units = {idx2id.get(i) for i in docket_to_count[dk] if idx2id.get(i)}
        real_units = set(docket_real[dk].keys())
        if not real_units and not llm_units: continue
        tp = len(llm_units & real_units)
        prec = tp / max(len(llm_units), 1) if llm_units else 0
        rec = tp / max(len(real_units), 1) if real_units else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        rows.append({'docket_id': dk, 'precision': prec, 'recall': rec, 'f1': f1})
        # accumulate attention
        all_units = set(idx2id.values()) | set(docket_real[dk].keys())
        for u in all_units:
            prov_idx = next((i for i, uid in idx2id.items() if uid == u), None)
            llm_attn.append(docket_to_count[dk].get(prov_idx, 0) if prov_idx else 0)
            real_attn.append(len(docket_real[dk].get(u, set())))
    df = pd.DataFrame(rows)
    real_arr = np.array(real_attn, dtype=float); llm_arr = np.array(llm_attn, dtype=float)
    pearson = float(np.corrcoef(real_arr, llm_arr)[0, 1]) if real_arr.sum() and llm_arr.sum() else 0.0

    return {
        'recall': float(df['recall'].mean()),
        'precision': float(df['precision'].mean()),
        'f1': float(df['f1'].mean()),
        'pearson_attention': pearson,
        'n_dockets_eval': len(rows),
    }


# ----------------- Axis B + C: pair labeling + frame labeling -----------------
# Opposing-sides framing improved judge accuracy 67% -> 96% (memory: feedback_llm_judge_opposing_sides).
PAIR_PROMPT = """\
You compare two regulatory comment claims (A and B) that target the SAME proposed regulatory provision. The two claims may come from commenters on opposing sides of the issue: agreeing, disagreeing, raising distinct concerns, or expressing the same point. Choose ONE label that best describes the relationship of B to A:

- same: B states the same point as A (a paraphrase or restatement; same direction).
- related: B is on the same sub-topic as A but makes a different point (still same direction or compatible).
- different: B addresses a different aspect/concern of the provision (no semantic overlap with A).
- opposing: B takes a contrary position to A on the same point (e.g., A wants extension, B wants no change).

Claim A: {a}
Claim B: {b}

Respond with one word: same, related, different, or opposing."""

FRAME_PROMPT = """\
Classify the dominant framing of this public-comment claim. Pick exactly ONE from this enum:
- health: public health, disease, medical outcomes
- safety: accidents, worker safety, product safety
- economic_cost_benefit: costs, jobs, competitiveness, ROI, market effects
- legal_constitutional: statutory authority, constitutional rights, APA, takings, due process
- scientific_technical: evidence, methodology, engineering, measurement
- environmental: pollution, ecosystems, climate, wildlife
- equity_justice: fairness, disparate impact, environmental justice, civil rights
- religious_moral: ethical or values-based appeals
- national_security: defense, foreign threats, critical infrastructure
- procedural: process objections, transparency, comment-period mechanics
- other
- unknown

Claim: {claim}

Respond with one word, the enum value."""

JUDGE_MODEL = 'gpt-5-mini'  # for axisB pair labels
FRAME_MODEL = 'gpt-5-mini'  # for axisC primary_frame labels


async def label_pair(client, sem, a, b):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'user', 'content': PAIR_PROMPT.format(a=a[:600], b=b[:600])}],
                max_completion_tokens=200, reasoning_effort='minimal',
                extra_body={'service_tier': 'flex'},
            )
            label = (resp.choices[0].message.content or '').strip().lower()
            label = label.split()[0] if label else 'unknown'
            in_tok = resp.usage.prompt_tokens; out_tok = resp.usage.completion_tokens
            return label, in_tok, out_tok
        except Exception:
            return 'error', 0, 0


async def label_frame(client, sem, claim):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=FRAME_MODEL,
                messages=[{'role': 'user', 'content': FRAME_PROMPT.format(claim=claim[:1500])}],
                max_completion_tokens=200, reasoning_effort='minimal',
                extra_body={'service_tier': 'flex'},
            )
            label = (resp.choices[0].message.content or '').strip().lower().split()[0]
            in_tok = resp.usage.prompt_tokens; out_tok = resp.usage.completion_tokens
            return label, in_tok, out_tok
        except Exception:
            return 'error', 0, 0


# ----------------- Axis C: JSD -----------------
FRAMES = ['environmental','health','equity_justice','economic_cost_benefit',
          'legal_constitutional','religious_moral','safety','scientific_technical',
          'procedural','national_security','other','unknown']
FRAME_MAP = {
    'public_health':'health', 'moral_religious':'religious_moral', 'moral':'religious_moral',
    'public_safety':'safety', 'consumer_protection':'other',
    'consumer':'other', 'consumer_impact':'other', 'transparency_accountability':'procedural',
    'animal_welfare':'environmental', 'human_rights':'equity_justice',
    'individual_rights':'equity_justice', 'humanitarian':'equity_justice',
    'cultural':'other', 'technical':'scientific_technical', 'professional_society':'other',
    'privacy':'legal_constitutional', 'educational':'other',
}

def normalize_frame(f):
    f = (f or '').strip().lower()
    if f in FRAMES: return f
    return FRAME_MAP.get(f, 'unknown')

def jsd(p, q):
    p = np.asarray(p, dtype=float) + 1e-12; q = np.asarray(q, dtype=float) + 1e-12
    p = p/p.sum(); q = q/q.sum()
    m = 0.5*(p+q)
    def kl(a,b): return float(np.sum(a*np.log(a/b)))
    return 0.5*kl(p,m) + 0.5*kl(q,m)


# ----------------- Axis B implementation -----------------
def _load_real_claims_for_benchmark(benchmark_dks):
    """Build dict: docket -> provision_idx -> list of (claim_id, claim_text). One row per
    Match2-YES (claim, unit_id). The same claim_id can appear under multiple provisions."""
    docket_units = load_units_per_docket()
    uid_to_idx = {dk: {u['global_id']: i+1 for i, u in enumerate(us)} for dk, us in docket_units.items()}
    real = defaultdict(lambda: defaultdict(list))
    for line in _open_text(MATCH2_LABELED):
        if '"llm_label": 1' not in line: continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get('llm_label') != 1: continue
        dk = r.get('docket_id')
        if dk not in benchmark_dks: continue
        uid = r.get('unit_id')
        prov_idx = uid_to_idx.get(dk, {}).get(uid)
        if prov_idx:
            real[dk][prov_idx].append((r.get('claim_id'), r.get('claim_text', '')))
    return real


_EMBED_MODEL = None
def _embed(texts, model_name='all-MiniLM-L6-v2'):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_name, device='cpu')
    return _EMBED_MODEL.encode(texts, batch_size=128, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)


async def compute_axis_b(output_dir, top_k=3, sample_dockets=None, concurrency=20, seed=13):
    """For each real Match2-YES claim, judge against the top-K most-similar model claims
    for the same (docket, provision). Strict = at least one 'same'; Loose = at least one
    'same' or 'related'."""
    runs = [json.loads(l) for l in open(output_dir / 'generated_claims.jsonl')]
    benchmark_dks = set(r['docket_id'] for r in runs)
    if sample_dockets:
        rng = random.Random(seed)
        keep = set(rng.sample(sorted(benchmark_dks), min(sample_dockets, len(benchmark_dks))))
        benchmark_dks = keep
        runs = [r for r in runs if r['docket_id'] in benchmark_dks]
        logger.info('axisB: sampled to %d dockets', len(benchmark_dks))

    # LLM claims grouped (dk, prov_idx) -> list of texts
    llm_idx = defaultdict(list)
    for r in runs:
        for c in r.get('parsed_claims', []):
            llm_idx[(r['docket_id'], c['provision_idx'])].append(c['claim_text'])

    real_idx = _load_real_claims_for_benchmark(benchmark_dks)
    # Build the work list: (real_text, [candidate_llm_texts])
    pairs_to_judge = []  # list of (real_claim_id, real_text, llm_text)
    real_claim_ids_seen = set()
    n_real_with_candidates = 0
    n_real_no_candidate = 0
    for dk, provs in real_idx.items():
        for prov, real_list in provs.items():
            llm_list = llm_idx.get((dk, prov), [])
            if not llm_list:
                for cid, _ in real_list:
                    real_claim_ids_seen.add(cid)
                    n_real_no_candidate += 1
                continue
            # Embed real + llm to find top-K candidates per real claim
            real_texts = [t for _, t in real_list]
            real_emb = _embed(real_texts)
            llm_emb = _embed(llm_list)
            sims = real_emb @ llm_emb.T  # (R, L)
            for i, (cid, rtext) in enumerate(real_list):
                real_claim_ids_seen.add(cid)
                n_real_with_candidates += 1
                top = np.argsort(-sims[i])[:top_k]
                for j in top:
                    pairs_to_judge.append((cid, rtext, llm_list[int(j)]))

    logger.info('axisB: %d real claims (%d with candidates, %d without), %d pair judgments',
                len(real_claim_ids_seen), n_real_with_candidates, n_real_no_candidate, len(pairs_to_judge))

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    in_tok = out_tok = 0
    pair_labels_path = output_dir / 'pair_judgments.jsonl'
    # Resume support: skip already-judged (cid, llm_text_hash) pairs
    done = set()
    if pair_labels_path.exists():
        for line in open(pair_labels_path):
            try:
                r = json.loads(line); done.add((r['claim_id'], r['llm_text_hash']))
            except Exception: pass
    if done:
        logger.info('axisB resume: %d pairs already judged', len(done))

    def _h(s): return str(hash(s))[:16]
    todo = [(cid, rt, lt) for cid, rt, lt in pairs_to_judge if (cid, _h(lt)) not in done]
    logger.info('axisB: judging %d new pairs (resumed %d)', len(todo), len(pairs_to_judge) - len(todo))

    results_by_cid = defaultdict(list)
    BATCH = 200
    with open(pair_labels_path, 'a') as fout:
        for s in range(0, len(todo), BATCH):
            batch = todo[s:s+BATCH]
            tasks = [label_pair(client, sem, rt, lt) for _, rt, lt in batch]
            outs = await asyncio.gather(*tasks)
            for (cid, rt, lt), (label, ti, to) in zip(batch, outs):
                in_tok += ti; out_tok += to
                results_by_cid[cid].append(label)
                fout.write(json.dumps({'claim_id': cid, 'llm_text_hash': _h(lt), 'label': label}) + '\n')
            fout.flush()
            elapsed = time.time() - t0
            logger.info('axisB %d/%d (%.0f/s) | tokens in=%d out=%d',
                        s + len(batch), len(todo), (s+len(batch))/max(elapsed,1), in_tok, out_tok)
    # Reload all labels (including resumed) to compute final metrics
    results_by_cid_all = defaultdict(list)
    for line in open(pair_labels_path):
        try:
            r = json.loads(line); results_by_cid_all[r['claim_id']].append(r['label'])
        except Exception: pass
    n_strict = n_loose = 0
    for cid in real_claim_ids_seen:
        labs = results_by_cid_all.get(cid, [])
        if any(l == 'same' for l in labs): n_strict += 1
        if any(l in ('same','related') for l in labs): n_loose += 1
    n_total = len(real_claim_ids_seen)
    return {
        'strict': n_strict / max(n_total, 1),
        'loose': n_loose / max(n_total, 1),
        'n_real_claims_evaluated': n_total,
        'n_real_claims_with_candidates': n_real_with_candidates,
        'n_pair_judgments': len(pairs_to_judge),
        'tokens_in': in_tok, 'tokens_out': out_tok,
    }


# ----------------- Axis C implementation -----------------
def _real_frame_path():
    return BENCH / 'real_claim_primary_frames_558.jsonl'


async def label_real_claims_for_benchmark(concurrency=50):
    """One-time labeling of real Match2-YES claims (deduped by claim_id) on benchmark
    dockets. Cached at data/derived/real_claim_frames.jsonl."""
    benchmark = load_benchmark_dockets()
    benchmark_dks = set(b['docket_id'] for b in benchmark)
    real_idx = _load_real_claims_for_benchmark(benchmark_dks)
    # Dedup by claim_id (use first-seen text)
    seen = {}
    for dk, provs in real_idx.items():
        for prov, real_list in provs.items():
            for cid, txt in real_list:
                if cid and cid not in seen: seen[cid] = (dk, txt)
    logger.info('axisC real: %d unique real claims to label', len(seen))

    out_path = _real_frame_path()
    done = {}
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line); done[r['claim_id']] = r['frame']
            except Exception: pass
    todo = [(cid, dk, txt) for cid, (dk, txt) in seen.items() if cid not in done]
    logger.info('axisC real: %d to label, %d cached', len(todo), len(done))
    if not todo:
        return done
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    sem = asyncio.Semaphore(concurrency)
    BATCH = 500
    with open(out_path, 'a') as fout:
        for s in range(0, len(todo), BATCH):
            batch = todo[s:s+BATCH]
            tasks = [label_frame(client, sem, txt) for _, _, txt in batch]
            outs = await asyncio.gather(*tasks)
            for (cid, dk, _), (frame, ti, to) in zip(batch, outs):
                fr = normalize_frame(frame)
                fout.write(json.dumps({'claim_id': cid, 'docket_id': dk, 'frame': fr}) + '\n')
                done[cid] = fr
            fout.flush()
            logger.info('axisC real %d/%d', s+len(batch), len(todo))
    return done


async def compute_axis_c(output_dir, concurrency=50):
    """Label LLM claims with primary_frame; compute per-docket JSD vs real distribution; mean."""
    runs = [json.loads(l) for l in open(output_dir / 'generated_claims.jsonl')]
    benchmark_dks = set(r['docket_id'] for r in runs)

    # Real frames (cached, possibly labeled now)
    real_frames = await label_real_claims_for_benchmark(concurrency=concurrency)
    # Build per-docket real frame distribution
    real_by_dk_frame = defaultdict(Counter)
    real_idx = _load_real_claims_for_benchmark(benchmark_dks)
    for dk, provs in real_idx.items():
        seen_cids = set()
        for prov, real_list in provs.items():
            for cid, _ in real_list:
                if not cid or cid in seen_cids: continue
                seen_cids.add(cid)
                fr = real_frames.get(cid)
                if fr: real_by_dk_frame[dk][fr] += 1

    # LLM frames: label them (cached per output_dir)
    llm_frame_path = output_dir / 'llm_claim_frames.jsonl'
    llm_done = {}
    if llm_frame_path.exists():
        for line in open(llm_frame_path):
            try:
                r = json.loads(line); llm_done[r['claim_key']] = r['frame']
            except Exception: pass
    # Build (claim_key, text) for all LLM claims
    llm_claims = []
    for r in runs:
        for ci, c in enumerate(r.get('parsed_claims', [])):
            key = f"{r['docket_id']}::{c['provision_idx']}::{c.get('stakeholder') or 'V'}::{ci}"
            llm_claims.append((key, r['docket_id'], c['claim_text']))
    todo = [(k, d, t) for k, d, t in llm_claims if k not in llm_done]
    logger.info('axisC LLM: %d to label, %d cached', len(todo), len(llm_done))
    if todo:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        sem = asyncio.Semaphore(concurrency)
        BATCH = 500
        with open(llm_frame_path, 'a') as fout:
            for s in range(0, len(todo), BATCH):
                batch = todo[s:s+BATCH]
                tasks = [label_frame(client, sem, t) for _, _, t in batch]
                outs = await asyncio.gather(*tasks)
                for (k, d, _), (frame, ti, to) in zip(batch, outs):
                    fr = normalize_frame(frame)
                    fout.write(json.dumps({'claim_key': k, 'docket_id': d, 'frame': fr}) + '\n')
                    llm_done[k] = fr
                fout.flush()
                logger.info('axisC LLM %d/%d', s+len(batch), len(todo))

    # Build per-docket LLM frame distribution
    llm_by_dk_frame = defaultdict(Counter)
    for k, d, _ in llm_claims:
        fr = llm_done.get(k)
        if fr: llm_by_dk_frame[d][fr] += 1

    # Compute JSD per docket and average (only dockets with both sides non-empty)
    jsds = []
    for dk in benchmark_dks:
        rc = real_by_dk_frame.get(dk, Counter())
        lc = llm_by_dk_frame.get(dk, Counter())
        if not rc or not lc: continue
        p = np.array([rc.get(f, 0) for f in FRAMES], dtype=float)
        q = np.array([lc.get(f, 0) for f in FRAMES], dtype=float)
        if p.sum() == 0 or q.sum() == 0: continue
        jsds.append(jsd(p, q))
    return {
        'jsd_mean': float(np.mean(jsds)) if jsds else 0.0,
        'jsd_median': float(np.median(jsds)) if jsds else 0.0,
        'n_dockets_eval': len(jsds),
        'n_llm_claims_labeled': len(llm_claims),
        'n_real_claims_labeled': len(real_frames),
    }


# ----------------- main runner -----------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='e.g. gpt-4o, gpt-5, claude-sonnet-4-5-20250929, meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--provider', required=True, choices=['openai', 'anthropic', 'vllm'])
    parser.add_argument('--base-url', default=None, help='for vllm: e.g. http://localhost:8001/v1')
    parser.add_argument('--variant', choices=['vanilla', 'subgroup'], default='vanilla')
    parser.add_argument('--output-name', required=True, help='subdir under data/derived/benchmark_runs/')
    parser.add_argument('--concurrency', type=int, default=20)
    parser.add_argument('--steps', default='gen,axisA',
                        help='comma-separated: gen, axisA, axisB, axisC, summary')
    parser.add_argument('--axisB-top-k', type=int, default=3,
                        help='top-K most-similar model claims to judge per real claim')
    parser.add_argument('--axisB-sample-dockets', type=int, default=None,
                        help='if set, sample this many dockets for axisB (for cost control)')
    parser.add_argument('--axisB-concurrency', type=int, default=50)
    parser.add_argument('--axisC-concurrency', type=int, default=50)
    args = parser.parse_args()

    output_dir = RUNS / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = set(args.steps.split(','))

    if 'gen' in steps:
        logger.info('=== Step 1: Generation ===')
        await run_generation(args.model, args.provider, args.variant, output_dir, args.concurrency,
                             base_url=args.base_url)

    # Load existing metrics if present so axes can be added incrementally
    metrics_path = output_dir / 'metrics.json'
    metrics = {}
    if metrics_path.exists():
        try: metrics = json.load(open(metrics_path))
        except Exception: metrics = {}

    if 'axisA' in steps:
        logger.info('=== Step 2: Axis A (provision targeting) ===')
        metrics.update({f'axisA_{k}': v for k, v in compute_axis_a(output_dir).items()})
        logger.info('Axis A: %s', metrics)

    if 'axisB' in steps:
        logger.info('=== Step 3: Axis B (semantic alignment) ===')
        b = await compute_axis_b(output_dir, top_k=args.axisB_top_k,
                                 sample_dockets=args.axisB_sample_dockets,
                                 concurrency=args.axisB_concurrency)
        metrics.update({f'axisB_{k}': v for k, v in b.items()})
        logger.info('Axis B: %s', b)

    if 'axisC' in steps:
        logger.info('=== Step 4: Axis C (frame distribution JSD) ===')
        c = await compute_axis_c(output_dir, concurrency=args.axisC_concurrency)
        metrics.update({f'axisC_{k}': v for k, v in c.items()})
        logger.info('Axis C: %s', c)

    if 'summary' in steps:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info('Saved %s', metrics_path)


if __name__ == '__main__':
    asyncio.run(main())
