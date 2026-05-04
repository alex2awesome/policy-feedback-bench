"""
utils.py — shared utilities used by the harness and the metric modules.

Contents:
  - File I/O helpers (gzip-aware open, JSONL iterator)
  - Benchmark data loaders (dockets, provisions, validated real claims)
  - Provision formatting + claim parsing
  - Sentence embedding helper (for finding semantic-nearest neighbors)
  - Frame normalization (maps loose strings to the canonical 12-class taxonomy)
  - Async LLM caller factory (OpenAI / Anthropic / vLLM-compatible servers)

None of these functions touch the API; importing this module is cheap and
side-effect-free.
"""
from __future__ import annotations
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from paths import (
    DOCKETS_FILE,
    PROVISIONS_FILE,
    REAL_CLAIMS_FILE,
)
from prompts import FRAMES, FRAME_ALIASES


# ============================================================================
# File I/O helpers
# ============================================================================

def open_text(path):
    """Open a text file for reading, transparently handling .gz files.

    Args:
        path: file path (str or Path). If the suffix is '.gz' the file
              is opened with gzip; otherwise opened as plain UTF-8 text.

    Returns:
        A text-mode file handle. Use as a context manager or iterate directly.
    """
    p = Path(path)
    return gzip.open(p, 'rt') if p.suffix == '.gz' else open(p)


def iter_jsonl(path) -> Iterator[dict]:
    """Yield each parsed JSON line of a JSONL file. Skips malformed lines.

    Args:
        path: file path; handled by open_text() so .gz is fine.

    Yields:
        dict for each line that successfully parses as JSON.
    """
    with open_text(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ============================================================================
# Benchmark data loaders
# ============================================================================

def load_benchmark_dockets() -> list[dict]:
    """Load the canonical 558-docket index for the benchmark.

    Returns:
        list of dicts, each with keys 'docket_id' and 'n_units'. Order is the
        order they appear in the underlying file (used as a stable iteration
        order during generation). Duplicates by docket_id are removed.
    """
    out = []
    seen = set()
    for r in iter_jsonl(DOCKETS_FILE):
        dk = r.get('docket_id')
        if dk and dk not in seen:
            seen.add(dk)
            out.append({'docket_id': dk, 'n_units': r.get('n_units')})
    return out


def load_provisions_per_docket() -> dict[str, list[dict]]:
    """Load all proposed-rule provisions (deontic units) for every benchmark docket.

    Each provision is a single normative statement extracted from the rule
    (an obligation, permission, prohibition, condition, or exception). Provisions
    keep the same order as in the underlying source file, which matters because
    the model's per-claim provision references (`[P1]`, `[P3]`, ...) are 1-indexed
    into this list per docket.

    Returns:
        dict mapping docket_id to a list of provision dicts. Each provision
        dict contains:
            - global_id: stable cross-docket identifier "docket::doc::unit"
            - section: CFR section reference if available (e.g. "§ 207.33(b)(2)")
            - type: one of {obligation, permission, prohibition, condition, exception}
            - summary: 1-sentence plain-language summary (truncated to 300 chars)
            - text: verbatim regulatory text (truncated to 800 chars)
    """
    units = defaultdict(list)
    for r in iter_jsonl(PROVISIONS_FILE):
        dk = r.get('docket_id')
        if not dk:
            continue
        doc_id = r.get('doc_id', '')
        for u in r.get('units', []):
            units[dk].append({
                'global_id': f"{dk}::{doc_id}::{u['id']}",
                'section': u.get('section', '') or '',
                'type': u.get('type', '') or '',
                'summary': (u.get('action_summary') or '')[:300],
                'text': (u.get('obligation_text') or '')[:800],
            })
    return dict(units)


def load_real_claims_per_provision(
    benchmark_docket_ids: set[str],
) -> dict[str, dict[int, list[tuple[str, str]]]]:
    """Load validated real human-comment claims, indexed by docket and provision.

    A "validated real claim" is a public-comment claim that has been matched
    to a specific proposed-rule provision via our matching pipeline (bi-encoder
    -> ModernBERT cross-encoder -> gpt-5-mini judge with opposing-sides framing)
    and confirmed positive ('llm_label' == 1). The same claim can be matched to
    multiple provisions, so the same claim_id may appear in multiple lists.

    Args:
        benchmark_docket_ids: restrict to claims belonging to these dockets.

    Returns:
        Nested dict:
            real[docket_id][provision_idx] = [(claim_id, claim_text), ...]
        where provision_idx is the 1-indexed position of the provision in the
        list returned by load_provisions_per_docket() for the same docket.
    """
    docket_units = load_provisions_per_docket()
    # Build (docket -> unit_global_id -> 1-indexed provision_idx) lookup.
    uid_to_idx = {
        dk: {u['global_id']: i + 1 for i, u in enumerate(us)}
        for dk, us in docket_units.items()
    }

    real: dict[str, dict[int, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in iter_jsonl(REAL_CLAIMS_FILE):
        if r.get('llm_label') != 1:
            continue
        dk = r.get('docket_id')
        if dk not in benchmark_docket_ids:
            continue
        prov_idx = uid_to_idx.get(dk, {}).get(r.get('unit_id'))
        if prov_idx:
            real[dk][prov_idx].append((r.get('claim_id'), r.get('claim_text', '')))
    return real


# ============================================================================
# Provision formatting + claim parsing (for prompts and model outputs)
# ============================================================================

def format_provisions_for_prompt(provisions: list[dict]) -> str:
    """Render the provision list as the bullet block that goes into the LLM prompt.

    Each provision is rendered as one line:
        P{i} [{section}] ({type}): {summary or first 300 chars of text}

    The 1-indexed `P{i}` is what the model uses to tag its output claims; the
    section and type provide context. Summary is preferred over verbatim text
    because it's shorter and more readable.

    Args:
        provisions: list of provision dicts as returned by load_provisions_per_docket.

    Returns:
        Newline-joined prompt-ready string.
    """
    lines = []
    for i, u in enumerate(provisions, 1):
        section_tag = f"[{u['section']}]" if u['section'] else ''
        body = u['summary'] or u['text'][:300]
        lines.append(f"P{i} {section_tag} ({u['type']}): {body}")
    return '\n'.join(lines)


# Match lines like "[P3] some claim text" — captures provision number and text.
_CLAIM_LINE_RE = re.compile(r'^\s*\[P(\d+)\]\s*(.+?)\s*$', re.MULTILINE)


def parse_claims_from_response(response_text: str, n_provisions: int) -> list[dict]:
    """Extract the model's tagged claims from its raw text response.

    Looks for lines matching `[P{int}] {claim text}`. Drops any claim whose
    provision index is outside the valid range [1, n_provisions]. This silently
    discards malformed model output (typically <2% of lines).

    Args:
        response_text: full model output (one model call).
        n_provisions: number of provisions for the docket; used to bound prov_idx.

    Returns:
        list of {'provision_idx': int, 'claim_text': str} dicts, in the order
        they appeared in the response.
    """
    out = []
    for m in _CLAIM_LINE_RE.finditer(response_text):
        idx = int(m.group(1))
        if 1 <= idx <= n_provisions:
            out.append({'provision_idx': idx, 'claim_text': m.group(2).strip()})
    return out


# ============================================================================
# Sentence embedding (used for axis-B candidate selection)
# ============================================================================

_EMBED_MODEL = None  # lazy-initialized SentenceTransformer (CPU)


def embed_texts(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Compute L2-normalized sentence embeddings for a batch of texts.

    Lazy-loads the SentenceTransformer on first call and reuses it across
    subsequent calls (the model load is the expensive part). Embeddings are
    normalized so cosine similarity is just `A @ B.T`.

    Args:
        texts: list of strings to embed.
        model_name: HuggingFace SentenceTransformer model name (default
                    'all-MiniLM-L6-v2', a small CPU-friendly model).

    Returns:
        numpy array of shape (len(texts), embedding_dim), dtype float32,
        L2-normalized.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_name, device='cpu')
    return _EMBED_MODEL.encode(
        texts,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


# ============================================================================
# Frame normalization
# ============================================================================

def normalize_frame(label: str | None) -> str:
    """Map an arbitrary frame string onto the canonical 12-class taxonomy.

    The judge model usually returns one of the 12 canonical labels in FRAMES,
    but it occasionally returns close variants (e.g. 'public_health' instead of
    'health', or 'technical' instead of 'scientific_technical'). FRAME_ALIASES
    handles these. Anything else falls through to 'unknown'.

    Args:
        label: a frame string, possibly None or with stray whitespace/casing.

    Returns:
        A member of FRAMES (one of 12 strings).
    """
    s = (label or '').strip().lower()
    if s in FRAMES:
        return s
    return FRAME_ALIASES.get(s, 'unknown')


# ============================================================================
# Jensen-Shannon divergence (used for axis C)
# ============================================================================

def jensen_shannon_divergence(p, q) -> float:
    """Compute JSD between two non-negative count vectors.

    JSD is symmetric and bounded in [0, ln 2]. 0 means the two distributions
    are identical; ln 2 (~0.693) means they have disjoint support. We add
    a tiny epsilon to avoid log(0) and normalize each input to a proper
    probability distribution.

    Args:
        p: array-like of non-negative counts (or probs).
        q: array-like of non-negative counts (or probs), same length as p.

    Returns:
        Scalar float JSD value.
    """
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ============================================================================
# Async LLM caller factory
# ============================================================================

async def _call_openai_chat(client, prompt, model, max_tokens=4000, reasoning_effort='low'):
    """Single OpenAI chat-completion call. Used for both OpenAI cloud and vLLM.

    For gpt-5-family models we automatically (a) set reasoning_effort and
    (b) request the flex service tier (50% cheaper, slightly slower). Other
    models get a vanilla chat-completion call.

    Args:
        client: openai.AsyncOpenAI instance.
        prompt: user-message string.
        model: model id (e.g. 'gpt-4o-mini', 'meta-llama/Llama-3.1-8B-Instruct').
        max_tokens: max completion tokens (default 4000).
        reasoning_effort: only used for gpt-5 family.

    Returns:
        Tuple (response_text, prompt_tokens, completion_tokens).
    """
    kwargs = dict(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        max_completion_tokens=max_tokens,
        extra_body={'service_tier': 'flex'} if 'gpt-5' in model else {},
    )
    if 'gpt-5' in model:
        kwargs['reasoning_effort'] = reasoning_effort
    resp = await client.chat.completions.create(**kwargs)
    return (
        (resp.choices[0].message.content or '').strip(),
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


async def _call_anthropic(client, prompt, model, max_tokens=4000):
    """Single Anthropic Messages API call.

    Args:
        client: anthropic.AsyncAnthropic instance.
        prompt: user-message string.
        model: Anthropic model id (e.g. 'claude-sonnet-4-5-20250929').
        max_tokens: max output tokens.

    Returns:
        Tuple (response_text, input_tokens, output_tokens).
    """
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = ''
    for block in resp.content:
        if hasattr(block, 'text'):
            text += block.text
    return text.strip(), resp.usage.input_tokens, resp.usage.output_tokens


def make_caller(
    provider: str,
    model: str,
    base_url: str | None = None,
) -> Callable[[str], Any]:
    """Build an async function `call(prompt) -> (text, in_tok, out_tok)`.

    Centralizes the provider-specific client setup so the rest of the code
    can ignore which API it's hitting.

    Args:
        provider: one of 'openai', 'anthropic', 'vllm'.
            - 'openai': uses os.environ['OPENAI_API_KEY']
            - 'anthropic': uses os.environ['ANTHROPIC_API_KEY']
            - 'vllm': hits a locally-served OpenAI-compatible endpoint
                      (api_key is irrelevant); requires base_url.
        model: model id passed through to the provider.
        base_url: only used for provider='vllm'; defaults to localhost:8001.

    Returns:
        An async function that takes a prompt string and returns
        (response_text, prompt_tokens, completion_tokens).

    Raises:
        ValueError: unknown provider.
    """
    if provider in ('openai', 'vllm'):
        from openai import AsyncOpenAI
        if provider == 'vllm':
            client = AsyncOpenAI(api_key='not-needed', base_url=base_url or 'http://localhost:8001/v1')
        else:
            client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])

        async def _call(prompt: str):
            return await _call_openai_chat(client, prompt, model)
        return _call

    elif provider == 'anthropic':
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

        async def _call(prompt: str):
            return await _call_anthropic(client, prompt, model)
        return _call

    raise ValueError(f'unknown provider: {provider!r} (expected one of openai, anthropic, vllm)')
