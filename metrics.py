"""
metrics.py — implementations for the four PolicyFeedbackBench evaluation axes.

Public entry points (called by run_benchmark.py):
  - compute_axis_a(output_dir): provision-targeting metrics; no LLM calls.
  - compute_axis_b(output_dir, ...): claim semantic alignment; uses LLM judge.
  - compute_axis_c(output_dir, ...): frame-distribution divergence; uses LLM labeler.

Each entry point returns a dict of metric values that can be merged into the
run's metrics.json summary.

axis-D (substantivity) is computed by a separate GPU-bound script; see
README §"Axis D substantivity reproducibility".
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from paths import (
    GENERATED_CLAIMS_FILENAME,
    LLM_CLAIM_FRAMES_FILENAME,
    PAIR_JUDGMENTS_FILENAME,
    REAL_CLAIM_FRAMES_FILE,
)
from prompts import FRAMES, FRAME_MODEL, JUDGE_MODEL, PAIR_PROMPT, FRAME_PROMPT
from utils import (
    embed_texts,
    iter_jsonl,
    jensen_shannon_divergence,
    load_benchmark_dockets,
    load_provisions_per_docket,
    load_real_claims_per_provision,
    normalize_frame,
    open_text,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Axis A — provision targeting (Recall, Precision, Pearson on attention shape)
# ============================================================================

def compute_axis_a(output_dir: Path) -> dict:
    """Score how well the model picks the same provisions to comment on as humans.

    For each docket:
      - True positive: a provision that BOTH real commenters and the model targeted
      - Recall = TP / (provisions real commenters targeted)
      - Precision = TP / (provisions the model targeted)

    We also compute Pearson correlation between per-provision real-claim counts
    and per-provision model-claim counts, pooled across all dockets. A high
    correlation means the model also matches the *intensity* (heavy-tailed shape)
    of human attention, not just the binary which-provisions-were-touched.

    Args:
        output_dir: per-run output directory containing generated_claims.jsonl.

    Returns:
        dict with keys: 'recall', 'precision', 'f1', 'pearson_attention',
        'n_dockets_eval'. The first three are per-docket means in [0, 1];
        Pearson is in [-1, 1]; n_dockets_eval is the count of dockets where
        either side targeted at least one provision.
    """
    runs = list(iter_jsonl(output_dir / GENERATED_CLAIMS_FILENAME))

    # Per docket: how many model claims targeted each provision_idx.
    docket_to_model_count = {
        r['docket_id']: _count_provisions_targeted(r.get('parsed_claims', []))
        for r in runs
    }

    # Build (docket -> 1-indexed provision_idx -> unit global_id) lookup.
    docket_units = load_provisions_per_docket()
    docket_to_idx_to_uid = {
        dk: {i + 1: u['global_id'] for i, u in enumerate(us)}
        for dk, us in docket_units.items()
    }

    # Real human attention: per docket, which unit_ids did real commenters touch?
    # (We only need the set of touched unit_ids for recall/precision; counts feed Pearson.)
    benchmark_dks = set(docket_to_model_count.keys())
    real_attention = _aggregate_real_attention(benchmark_dks)

    # Per-docket recall/precision/f1 + accumulate vectors for Pearson.
    rows = []
    real_counts_vec, model_counts_vec = [], []
    for dk in benchmark_dks:
        idx_to_uid = docket_to_idx_to_uid.get(dk, {})
        model_units = {idx_to_uid.get(i) for i in docket_to_model_count[dk] if idx_to_uid.get(i)}
        real_units = set(real_attention[dk].keys())
        if not model_units and not real_units:
            continue
        tp = len(model_units & real_units)
        prec = tp / max(len(model_units), 1) if model_units else 0.0
        rec = tp / max(len(real_units), 1) if real_units else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        rows.append({'docket_id': dk, 'precision': prec, 'recall': rec, 'f1': f1})

        # Pearson: per-provision count vector (real vs model) over the union of every
        # provision in the docket plus any provision with real attention. We include
        # zero-zero entries (provisions neither side touched) so the correlation reflects
        # the full per-provision distribution, not just touched ones.
        all_units_in_docket = set(idx_to_uid.values()) | set(real_attention[dk].keys())
        for uid in all_units_in_docket:
            prov_idx = next((i for i, u in idx_to_uid.items() if u == uid), None)
            model_counts_vec.append(docket_to_model_count[dk].get(prov_idx, 0) if prov_idx else 0)
            real_counts_vec.append(len(real_attention[dk].get(uid, set())))

    df = pd.DataFrame(rows)
    real_arr = np.array(real_counts_vec, dtype=float)
    model_arr = np.array(model_counts_vec, dtype=float)
    pearson = (
        float(np.corrcoef(real_arr, model_arr)[0, 1])
        if real_arr.sum() and model_arr.sum() else 0.0
    )

    return {
        'recall': float(df['recall'].mean()),
        'precision': float(df['precision'].mean()),
        'f1': float(df['f1'].mean()),
        'pearson_attention': pearson,
        'n_dockets_eval': len(rows),
    }


def _count_provisions_targeted(parsed_claims: list[dict]) -> Counter:
    """Tally how many model claims target each 1-indexed provision."""
    counts = Counter()
    for cl in parsed_claims:
        idx = cl.get('provision_idx')
        if idx:
            counts[idx] += 1
    return counts


def _aggregate_real_attention(
    benchmark_dks: set[str],
) -> dict[str, dict[str, set[str]]]:
    """Build docket -> unit_id -> set(cluster_uid) over validated real claims."""
    real = defaultdict(lambda: defaultdict(set))
    real_idx = load_real_claims_per_provision(benchmark_dks)
    # load_real_claims_per_provision indexes by (dk, prov_idx); we need it by (dk, unit_id).
    # Rebuild from the inner data: use the docket_units mapping.
    docket_units = load_provisions_per_docket()
    for dk in benchmark_dks:
        units = docket_units.get(dk, [])
        idx_to_uid = {i + 1: u['global_id'] for i, u in enumerate(units)}
        for prov_idx, claims in real_idx.get(dk, {}).items():
            uid = idx_to_uid.get(prov_idx)
            if not uid:
                continue
            for cid, _ in claims:
                if cid:
                    # cluster_uid = first segment of claim_id "DOCKET::CLUSTER__claim_idx".
                    cluster_uid = cid.split('__')[0] if '__' in cid else cid
                    real[dk][uid].add(cluster_uid)
    return real


# ============================================================================
# Axis B — claim semantic alignment (LLM judge)
# ============================================================================

async def compute_axis_b(
    output_dir: Path,
    top_k: int = 3,
    sample_dockets: int | None = None,
    concurrency: int = 20,
    seed: int = 13,
) -> dict:
    """Score how often the model paraphrases real human commenters.

    Algorithm overview:
      1. For each (docket, provision) where both real and model claims exist,
         embed both sides and pick the top-K most-similar model claims per
         real claim.
      2. Send each (real, model) pair to the LLM judge; collect labels.
      3. For each unique real claim, mark it 'strict-aligned' if at least one
         of its judged candidates was labeled 'same'; 'loose-aligned' if at
         least one was 'same' or 'related'.
      4. Report the fraction of real claims that are strict / loose aligned.

    Args:
        output_dir: per-run directory; reads generated_claims.jsonl, writes
                    pair_judgments.jsonl. Resume-safe: re-running picks up
                    where it left off based on (claim_id, llm_text_hash).
        top_k: number of top-similarity model candidates to judge per real
               claim (default 3). Smaller = cheaper but possibly under-counts.
        sample_dockets: if set, restrict the analysis to a random sample of
                        this many dockets (for cost control).
        concurrency: max simultaneous in-flight judge calls.
        seed: random seed for sample_dockets sampling (for reproducibility).

    Returns:
        dict with strict/loose rates plus counts and token usage.
    """
    runs = list(iter_jsonl(output_dir / GENERATED_CLAIMS_FILENAME))
    benchmark_dks = _maybe_sample_dockets(
        all_dks=set(r['docket_id'] for r in runs),
        sample_size=sample_dockets,
        seed=seed,
    )
    runs = [r for r in runs if r['docket_id'] in benchmark_dks]

    pairs_to_judge, real_claim_ids_seen, n_with_candidates = _build_judge_workload(
        runs=runs,
        benchmark_dks=benchmark_dks,
        top_k=top_k,
    )
    logger.info(
        'axisB: %d unique real claims (%d with candidates), %d total pairs to judge',
        len(real_claim_ids_seen), n_with_candidates, len(pairs_to_judge),
    )

    pair_labels_path = output_dir / PAIR_JUDGMENTS_FILENAME
    in_tok, out_tok = await _run_judge_pass(
        pairs_to_judge=pairs_to_judge,
        out_path=pair_labels_path,
        concurrency=concurrency,
    )

    strict_rate, loose_rate = _aggregate_strict_loose(
        pair_labels_path=pair_labels_path,
        real_claim_ids=real_claim_ids_seen,
    )
    return {
        'strict': strict_rate,
        'loose': loose_rate,
        'n_real_claims_evaluated': len(real_claim_ids_seen),
        'n_real_claims_with_candidates': n_with_candidates,
        'n_pair_judgments': len(pairs_to_judge),
        'tokens_in': in_tok,
        'tokens_out': out_tok,
    }


def _maybe_sample_dockets(all_dks: set[str], sample_size: int | None, seed: int) -> set[str]:
    """If sample_size is set, randomly sample that many dockets; else return all."""
    if not sample_size:
        return all_dks
    rng = random.Random(seed)
    keep = set(rng.sample(sorted(all_dks), min(sample_size, len(all_dks))))
    logger.info('axisB: sampled to %d dockets', len(keep))
    return keep


def _build_judge_workload(
    runs: list[dict],
    benchmark_dks: set[str],
    top_k: int,
) -> tuple[list[tuple[str, str, str]], set[str], int]:
    """Decide which (real_claim, model_claim) pairs to send to the judge.

    For each (docket, provision_idx):
      - Get the validated real claims for that provision
      - Get the model claims for that provision
      - If model has zero claims on this provision, all real claims here are "no candidate"
      - Otherwise, embed both sides and keep the top-K model claims per real claim

    Returns:
        Tuple of (pairs_to_judge, real_claim_ids_seen, n_with_candidates):
            - pairs_to_judge: list of (real_claim_id, real_text, model_text) triples
            - real_claim_ids_seen: every real claim_id encountered (for denominator)
            - n_with_candidates: count of real claims that had at least one model candidate
    """
    # Index model claims by (docket, provision_idx).
    model_claims_by_prov: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in runs:
        for c in r.get('parsed_claims', []):
            model_claims_by_prov[(r['docket_id'], c['provision_idx'])].append(c['claim_text'])

    real_idx = load_real_claims_per_provision(benchmark_dks)
    pairs: list[tuple[str, str, str]] = []
    real_claim_ids_seen: set[str] = set()
    n_with_candidates = 0

    for dk, provs in real_idx.items():
        for prov_idx, real_list in provs.items():
            real_claim_ids_seen.update(cid for cid, _ in real_list if cid)
            model_texts = model_claims_by_prov.get((dk, prov_idx), [])
            if not model_texts:
                continue  # no candidates -> these real claims will count as not-aligned

            real_texts = [t for _, t in real_list]
            real_emb = embed_texts(real_texts)
            model_emb = embed_texts(model_texts)
            similarity = real_emb @ model_emb.T  # (R, M)

            for i, (cid, rtext) in enumerate(real_list):
                n_with_candidates += 1
                top_indices = np.argsort(-similarity[i])[:top_k]
                for j in top_indices:
                    pairs.append((cid, rtext, model_texts[int(j)]))

    return pairs, real_claim_ids_seen, n_with_candidates


def _hash_text(text: str) -> str:
    """Short stable hash of a model-claim string (for resume deduplication)."""
    return str(hash(text))[:16]


async def _run_judge_pass(
    pairs_to_judge: list[tuple[str, str, str]],
    out_path: Path,
    concurrency: int,
    batch_size: int = 200,
) -> tuple[int, int]:
    """Send each (real, model) pair to the LLM judge; append results to out_path.

    Resume-safe: if out_path already exists, pairs whose (claim_id, model_text_hash)
    is already recorded are skipped.

    Args:
        pairs_to_judge: list of (real_claim_id, real_text, model_text) triples.
        out_path: JSONL file to append judgments to.
        concurrency: max simultaneous in-flight calls.
        batch_size: how many pairs to dispatch per asyncio.gather batch
                    (controls flush granularity / progress logging).

    Returns:
        (input_tokens_total, output_tokens_total) consumed by judge calls.
    """
    done_keys = _load_completed_judgments(out_path)
    if done_keys:
        logger.info('axisB resume: %d pairs already judged', len(done_keys))

    todo = [(cid, rt, mt) for cid, rt, mt in pairs_to_judge
            if (cid, _hash_text(mt)) not in done_keys]
    logger.info('axisB: judging %d new pairs (resumed %d)',
                len(todo), len(pairs_to_judge) - len(todo))

    if not todo:
        return 0, 0

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    sem = asyncio.Semaphore(concurrency)
    in_tok = out_tok = 0
    t0 = time.time()

    with open(out_path, 'a') as fout:
        for s in range(0, len(todo), batch_size):
            batch = todo[s:s + batch_size]
            tasks = [_label_pair(client, sem, rt, mt) for _, rt, mt in batch]
            outs = await asyncio.gather(*tasks)
            for (cid, _rt, mt), (label, ti, to) in zip(batch, outs):
                in_tok += ti
                out_tok += to
                fout.write(json.dumps({
                    'claim_id': cid,
                    'llm_text_hash': _hash_text(mt),
                    'label': label,
                }) + '\n')
            fout.flush()
            elapsed = time.time() - t0
            n_done = s + len(batch)
            logger.info('axisB %d/%d (%.0f/s) | tokens in=%d out=%d',
                        n_done, len(todo), n_done / max(elapsed, 1), in_tok, out_tok)

    return in_tok, out_tok


def _load_completed_judgments(out_path: Path) -> set[tuple[str, str]]:
    """Read existing judgments from out_path; return set of (claim_id, hash) keys."""
    done = set()
    if not out_path.exists():
        return done
    for r in iter_jsonl(out_path):
        if 'claim_id' in r and 'llm_text_hash' in r:
            done.add((r['claim_id'], r['llm_text_hash']))
    return done


def _aggregate_strict_loose(
    pair_labels_path: Path,
    real_claim_ids: set[str],
) -> tuple[float, float]:
    """Compute strict and loose alignment rates from the saved pair judgments.

    Args:
        pair_labels_path: JSONL file of judge labels (one row per pair).
        real_claim_ids: full set of real claim_ids that should be in the
                        denominator (includes claims with no candidates;
                        those count as not aligned by either metric).

    Returns:
        (strict_rate, loose_rate). Both in [0, 1].
    """
    labels_by_cid = defaultdict(list)
    if pair_labels_path.exists():
        for r in iter_jsonl(pair_labels_path):
            labels_by_cid[r['claim_id']].append(r.get('label'))

    n_strict = sum(
        1 for cid in real_claim_ids
        if any(l == 'same' for l in labels_by_cid.get(cid, []))
    )
    n_loose = sum(
        1 for cid in real_claim_ids
        if any(l in ('same', 'related') for l in labels_by_cid.get(cid, []))
    )
    n_total = max(len(real_claim_ids), 1)
    return n_strict / n_total, n_loose / n_total


async def _label_pair(client, sem, claim_a: str, claim_b: str) -> tuple[str, int, int]:
    """Single judge call: label one (claim A, claim B) pair.

    Returns ('same' | 'related' | 'different' | 'opposing' | 'error',
             prompt_tokens, completion_tokens). On any exception we return
    ('error', 0, 0) so the caller can keep going.
    """
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'user',
                           'content': PAIR_PROMPT.format(a=claim_a[:600], b=claim_b[:600])}],
                max_completion_tokens=200,
                reasoning_effort='minimal',
                extra_body={'service_tier': 'flex'},
            )
            label = (resp.choices[0].message.content or '').strip().lower()
            label = label.split()[0] if label else 'unknown'
            return label, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception:
            return 'error', 0, 0


# ============================================================================
# Axis C — frame-distribution divergence (LLM labels + JSD)
# ============================================================================

async def compute_axis_c(
    output_dir: Path,
    concurrency: int = 50,
) -> dict:
    """Score how well the model's topical mix matches real commenters' mix.

    For each docket:
      - Build the real-claim primary-frame distribution (12-class taxonomy)
      - Build the model-claim primary-frame distribution
      - Compute Jensen-Shannon divergence between them

    Frames are labeled by an LLM (gpt-5-mini by default). Real-claim labels
    are shipped pre-computed at REAL_CLAIM_FRAMES_FILE; model-claim labels
    are computed on the fly and cached per run.

    Args:
        output_dir: per-run directory; reads generated_claims.jsonl, writes
                    llm_claim_frames.jsonl.
        concurrency: max simultaneous in-flight labeler calls.

    Returns:
        dict with mean and median per-docket JSD plus counts.
    """
    runs = list(iter_jsonl(output_dir / GENERATED_CLAIMS_FILENAME))
    benchmark_dks = set(r['docket_id'] for r in runs)

    # Real-claim frame labels (pre-computed cache; one-time labeling step lives elsewhere).
    real_frames = _load_real_claim_frames()
    real_per_docket = _per_docket_frame_distribution_real(benchmark_dks, real_frames)

    # Model-claim frame labels (compute now, cache per run).
    model_claims = _enumerate_model_claims(runs)
    llm_done = await _label_model_claims(
        model_claims=model_claims,
        out_path=output_dir / LLM_CLAIM_FRAMES_FILENAME,
        concurrency=concurrency,
    )
    model_per_docket = _per_docket_frame_distribution_model(model_claims, llm_done)

    jsds = _per_docket_jsds(benchmark_dks, real_per_docket, model_per_docket)

    return {
        'jsd_mean': float(np.mean(jsds)) if jsds else 0.0,
        'jsd_median': float(np.median(jsds)) if jsds else 0.0,
        'n_dockets_eval': len(jsds),
        'n_llm_claims_labeled': len(model_claims),
        'n_real_claims_labeled': len(real_frames),
    }


def _load_real_claim_frames() -> dict[str, str]:
    """Load the pre-computed real-claim primary_frame labels (claim_id -> frame).

    Returns an empty dict if the file is missing — in that case we'd need to
    label the real claims live, which is outside this function's scope.
    """
    out = {}
    if not REAL_CLAIM_FRAMES_FILE.exists():
        logger.warning('No cached real_claim_frames file at %s; axis C will under-report.',
                       REAL_CLAIM_FRAMES_FILE)
        return out
    for r in iter_jsonl(REAL_CLAIM_FRAMES_FILE):
        if r.get('claim_id') and r.get('frame'):
            out[r['claim_id']] = r['frame']
    return out


def _per_docket_frame_distribution_real(
    benchmark_dks: set[str],
    real_frames: dict[str, str],
) -> dict[str, Counter]:
    """Per-docket frame Counter over unique validated real claims."""
    real_per_docket = defaultdict(Counter)
    real_idx = load_real_claims_per_provision(benchmark_dks)
    for dk, provs in real_idx.items():
        seen_cids = set()
        for prov_idx, real_list in provs.items():
            for cid, _ in real_list:
                if not cid or cid in seen_cids:
                    continue
                seen_cids.add(cid)
                fr = real_frames.get(cid)
                if fr:
                    real_per_docket[dk][fr] += 1
    return real_per_docket


def _enumerate_model_claims(runs: list[dict]) -> list[tuple[str, str, str]]:
    """Build a flat list of (claim_key, docket_id, claim_text) for every model claim.

    claim_key uniquely identifies the claim within a run (used for resume).
    Stakeholder is included in the key so vanilla and persona claims for the
    same provision don't collide.
    """
    out = []
    for r in runs:
        for ci, c in enumerate(r.get('parsed_claims', [])):
            stakeholder = c.get('stakeholder') or 'V'  # 'V' = vanilla
            key = f"{r['docket_id']}::{c['provision_idx']}::{stakeholder}::{ci}"
            out.append((key, r['docket_id'], c['claim_text']))
    return out


async def _label_model_claims(
    model_claims: list[tuple[str, str, str]],
    out_path: Path,
    concurrency: int,
    batch_size: int = 500,
) -> dict[str, str]:
    """Label every model claim's primary_frame; cache results to out_path.

    Resume-safe: claims already in out_path are skipped.

    Returns:
        dict mapping claim_key -> normalized frame label.
    """
    done = {}
    if out_path.exists():
        for r in iter_jsonl(out_path):
            if r.get('claim_key') and r.get('frame'):
                done[r['claim_key']] = r['frame']

    todo = [(k, dk, t) for k, dk, t in model_claims if k not in done]
    logger.info('axisC LLM: %d model claims to label, %d cached', len(todo), len(done))
    if not todo:
        return done

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    sem = asyncio.Semaphore(concurrency)
    with open(out_path, 'a') as fout:
        for s in range(0, len(todo), batch_size):
            batch = todo[s:s + batch_size]
            tasks = [_label_frame(client, sem, t) for _, _, t in batch]
            outs = await asyncio.gather(*tasks)
            for (k, dk, _), (frame, _ti, _to) in zip(batch, outs):
                fr = normalize_frame(frame)
                fout.write(json.dumps({'claim_key': k, 'docket_id': dk, 'frame': fr}) + '\n')
                done[k] = fr
            fout.flush()
            logger.info('axisC LLM %d/%d', s + len(batch), len(todo))
    return done


def _per_docket_frame_distribution_model(
    model_claims: list[tuple[str, str, str]],
    llm_labels: dict[str, str],
) -> dict[str, Counter]:
    """Per-docket frame Counter over all model claims for that docket."""
    per_docket = defaultdict(Counter)
    for k, dk, _ in model_claims:
        fr = llm_labels.get(k)
        if fr:
            per_docket[dk][fr] += 1
    return per_docket


def _per_docket_jsds(
    benchmark_dks: set[str],
    real_per_docket: dict[str, Counter],
    model_per_docket: dict[str, Counter],
) -> list[float]:
    """Compute per-docket JSD vector (only for dockets where both sides are non-empty)."""
    out = []
    for dk in benchmark_dks:
        real_counter = real_per_docket.get(dk, Counter())
        model_counter = model_per_docket.get(dk, Counter())
        if not real_counter or not model_counter:
            continue
        p = np.array([real_counter.get(f, 0) for f in FRAMES], dtype=float)
        q = np.array([model_counter.get(f, 0) for f in FRAMES], dtype=float)
        if p.sum() == 0 or q.sum() == 0:
            continue
        out.append(jensen_shannon_divergence(p, q))
    return out


async def _label_frame(client, sem, claim_text: str) -> tuple[str, int, int]:
    """Single labeler call: assign a primary_frame to one claim.

    Returns (frame_label, prompt_tokens, completion_tokens). Returns
    ('error', 0, 0) on failure so the caller can continue.
    """
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=FRAME_MODEL,
                messages=[{'role': 'user',
                           'content': FRAME_PROMPT.format(claim=claim_text[:1500])}],
                max_completion_tokens=200,
                reasoning_effort='minimal',
                extra_body={'service_tier': 'flex'},
            )
            label = (resp.choices[0].message.content or '').strip().lower().split()[0]
            return label, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception:
            return 'error', 0, 0
