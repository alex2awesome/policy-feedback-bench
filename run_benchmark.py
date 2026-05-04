"""
run_benchmark.py — entry point for evaluating a single language model on PolicyFeedbackBench.

Pipeline (each step has an --steps name; steps are independent and resumable):

    gen      Generate predicted public-comment claims for each of the 558 dockets
             (writes generated_claims.jsonl).
    axisA    Score provision-targeting (recall, precision, Pearson on attention shape).
    axisB    Score claim semantic alignment via LLM judge (writes pair_judgments.jsonl,
             reports Strict and Loose match rates).
    axisC    Score frame-distribution divergence via LLM labeler (writes
             llm_claim_frames.jsonl, reports JSD vs real distribution).
    summary  Merge whatever metrics were just computed into metrics.json.

Typical usage:

    # Full run on a frontier OpenAI model, vanilla prompting
    python run_benchmark.py \\
        --model gpt-4o-mini --provider openai \\
        --variant vanilla \\
        --output-name gpt-4o-mini_vanilla \\
        --steps gen,axisA,axisB,axisC,summary

    # Local open-weight model via vLLM, persona-conditioned prompting
    python run_benchmark.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --provider vllm --base-url http://localhost:8001/v1 \\
        --variant subgroup \\
        --output-name llama-3.1-8b-instruct_subgroup \\
        --steps gen,axisA,axisB,axisC,summary

See README.md for full metric definitions and cost-control flags.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

from paths import GENERATED_CLAIMS_FILENAME, METRICS_FILENAME, RUNS_DIR
from prompts import GEN_PROMPT_SUBGROUP, GEN_PROMPT_VANILLA, STAKEHOLDER_PERSONAS
from utils import (
    format_provisions_for_prompt,
    iter_jsonl,
    load_benchmark_dockets,
    load_provisions_per_docket,
    make_caller,
    parse_claims_from_response,
)
from metrics import compute_axis_a, compute_axis_b, compute_axis_c

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Generation step
# ============================================================================

CallFn = Callable[[str], Awaitable[tuple[str, int, int]]]


async def _one_call(
    call: CallFn,
    sem: asyncio.Semaphore,
    prompt: str,
    docket_id: str,
    persona_tag: str,
) -> tuple[str, int, int, str]:
    """Make one model call, holding a semaphore slot for the duration.

    Wraps the API call in a try/except so a single failure does not abort
    the rest of a batch. Logs failures so they can be diagnosed.

    Args:
        call: async function returned by make_caller().
        sem: shared semaphore that bounds total in-flight calls.
        prompt: user-message string.
        docket_id: included in the failure log so we can trace back.
        persona_tag: 'vanilla' or one of the seven STAKEHOLDER_PERSONAS keys.

    Returns:
        (response_text, prompt_tokens, completion_tokens, persona_tag).
        On failure: ('', 0, 0, persona_tag).
    """
    async with sem:
        try:
            text, in_tok, out_tok = await call(prompt)
            return text, in_tok, out_tok, persona_tag
        except Exception as e:
            logger.warning('call failed for %s/%s: %s', docket_id, persona_tag, str(e)[:120])
            return '', 0, 0, persona_tag


async def generate_for_docket(
    call: CallFn,
    docket_id: str,
    provisions: list[dict],
    variant: str,
    sem: asyncio.Semaphore,
) -> tuple[list[dict], int, int]:
    """Run the model for one docket. Returns parsed claims + token usage.

    For variant='vanilla': one model call per docket.
    For variant='subgroup': seven concurrent calls (one per stakeholder
    persona); their outputs are pooled, with each claim tagged with the
    persona that produced it.

    Args:
        call: async function from make_caller().
        docket_id: docket identifier.
        provisions: list of provision dicts (from load_provisions_per_docket).
        variant: 'vanilla' or 'subgroup'.
        sem: semaphore that bounds total in-flight calls across all dockets.

    Returns:
        (claims, prompt_tokens_total, completion_tokens_total) where claims is
        a list of {'provision_idx': int, 'claim_text': str, 'stakeholder': str|None}.
    """
    provisions_str = format_provisions_for_prompt(provisions)
    n_provisions = len(provisions)
    all_claims: list[dict] = []
    in_tok_total = out_tok_total = 0

    if variant == 'vanilla':
        prompt = GEN_PROMPT_VANILLA.format(provisions=provisions_str)
        text, in_tok, out_tok, _ = await _one_call(
            call, sem, prompt, docket_id, persona_tag='vanilla',
        )
        in_tok_total += in_tok
        out_tok_total += out_tok
        for c in parse_claims_from_response(text, n_provisions):
            c['stakeholder'] = None
            all_claims.append(c)
        return all_claims, in_tok_total, out_tok_total

    # variant == 'subgroup': fire all 7 persona prompts concurrently.
    tasks = []
    for persona_key, (label, persona_desc) in STAKEHOLDER_PERSONAS.items():
        persona_prompt = GEN_PROMPT_SUBGROUP.format(
            provisions=provisions_str,
            stakeholder_persona=persona_desc,
            stakeholder_label=label,
        )
        tasks.append(_one_call(call, sem, persona_prompt, docket_id, persona_tag=persona_key))
    results = await asyncio.gather(*tasks)
    for text, in_tok, out_tok, persona_key in results:
        in_tok_total += in_tok
        out_tok_total += out_tok
        for c in parse_claims_from_response(text, n_provisions):
            c['stakeholder'] = persona_key
            all_claims.append(c)
    return all_claims, in_tok_total, out_tok_total


async def run_generation(
    model: str,
    provider: str,
    variant: str,
    output_dir: Path,
    concurrency: int = 20,
    base_url: str | None = None,
    batch_size: int = 20,
) -> None:
    """Generate model claims for every benchmark docket, with resume support.

    Writes one JSONL line per docket to generated_claims.jsonl. If that file
    already exists, dockets that have a line in it are skipped (so killing
    and re-running picks up where you left off).

    Args:
        model: model id (e.g. 'gpt-4o-mini', 'meta-llama/Llama-3.1-8B-Instruct').
        provider: 'openai' | 'anthropic' | 'vllm'.
        variant: 'vanilla' | 'subgroup'.
        output_dir: per-run directory; will be created if missing.
        concurrency: max simultaneous API calls.
        base_url: only for provider='vllm'.
        batch_size: how many dockets to fire per asyncio.gather() round.
                    Affects flush granularity / progress logging only.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / GENERATED_CLAIMS_FILENAME
    call = make_caller(provider, model, base_url=base_url)

    benchmark = load_benchmark_dockets()
    logger.info('Loaded %d benchmark dockets', len(benchmark))
    docket_provisions = load_provisions_per_docket()

    done_dks = {r.get('docket_id') for r in iter_jsonl(out_path)} if out_path.exists() else set()
    todo = [b for b in benchmark if b['docket_id'] not in done_dks]
    logger.info('Resume: %d already generated, %d remaining', len(done_dks), len(todo))

    sem = asyncio.Semaphore(concurrency)
    in_tok_total = out_tok_total = 0
    t0 = time.time()
    with open(out_path, 'a') as fout:
        for s in range(0, len(todo), batch_size):
            batch = todo[s:s + batch_size]
            tasks = [
                generate_for_docket(call, b['docket_id'], docket_provisions[b['docket_id']],
                                    variant, sem)
                for b in batch
            ]
            results = await asyncio.gather(*tasks)
            for b, (claims, in_tok, out_tok) in zip(batch, results):
                fout.write(json.dumps({
                    'docket_id': b['docket_id'],
                    'n_units': b['n_units'],
                    'parsed_claims': claims,
                    'variant': variant,
                    'model': model,
                }) + '\n')
                in_tok_total += in_tok
                out_tok_total += out_tok
            fout.flush()
            elapsed = time.time() - t0
            n_done = s + len(batch)
            logger.info('%d/%d (%.1f/s) | tokens in=%d out=%d',
                        n_done, len(todo), n_done / max(elapsed, 1), in_tok_total, out_tok_total)
    logger.info('Done: %s | in=%d out=%d', out_path, in_tok_total, out_tok_total)


# ============================================================================
# CLI entry point
# ============================================================================

def _parse_args() -> argparse.Namespace:
    """Define and parse the harness command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True,
                   help="Model id, e.g. 'gpt-4o-mini', 'gpt-5', "
                        "'claude-sonnet-4-5-20250929', 'meta-llama/Llama-3.1-8B-Instruct'")
    p.add_argument('--provider', required=True, choices=['openai', 'anthropic', 'vllm'])
    p.add_argument('--base-url', default=None,
                   help='For provider=vllm: e.g. http://localhost:8001/v1')
    p.add_argument('--variant', choices=['vanilla', 'subgroup'], default='vanilla',
                   help='vanilla = 1 prompt per docket; subgroup = 7 stakeholder personas pooled')
    p.add_argument('--output-name', required=True,
                   help='Subdirectory name under reference_runs/ where outputs land')
    p.add_argument('--concurrency', type=int, default=20,
                   help='Max simultaneous in-flight generation calls')
    p.add_argument('--steps', default='gen,axisA',
                   help="Comma-separated subset of {gen, axisA, axisB, axisC, summary}")
    p.add_argument('--axisB-top-k', type=int, default=3,
                   help='Top-K most-similar model claims to judge per real claim '
                        '(higher = more accurate but more expensive)')
    p.add_argument('--axisB-sample-dockets', type=int, default=None,
                   help='If set, restrict axis B to a random sample of N dockets '
                        '(axes A and C still use the full 558)')
    p.add_argument('--axisB-concurrency', type=int, default=50,
                   help='Max simultaneous in-flight judge calls (axis B)')
    p.add_argument('--axisC-concurrency', type=int, default=50,
                   help='Max simultaneous in-flight labeler calls (axis C)')
    return p.parse_args()


def _load_existing_metrics(metrics_path: Path) -> dict:
    """Return the metrics.json contents if present, else empty dict.

    Lets us add or update individual axis metrics without losing earlier ones.
    """
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        logger.warning('Could not parse existing %s; starting fresh.', metrics_path)
        return {}


async def main() -> None:
    """Top-level orchestrator: parse args, run requested steps, write metrics.json."""
    args = _parse_args()

    output_dir = RUNS_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_steps = set(args.steps.split(','))

    if 'gen' in requested_steps:
        logger.info('=== Step 1: Generation ===')
        await run_generation(
            model=args.model,
            provider=args.provider,
            variant=args.variant,
            output_dir=output_dir,
            concurrency=args.concurrency,
            base_url=args.base_url,
        )

    metrics_path = output_dir / METRICS_FILENAME
    metrics = _load_existing_metrics(metrics_path)

    if 'axisA' in requested_steps:
        logger.info('=== Step 2: Axis A (provision targeting) ===')
        a = compute_axis_a(output_dir)
        metrics.update({f'axisA_{k}': v for k, v in a.items()})
        logger.info('Axis A: %s', a)

    if 'axisB' in requested_steps:
        logger.info('=== Step 3: Axis B (claim semantic alignment) ===')
        b = await compute_axis_b(
            output_dir=output_dir,
            top_k=args.axisB_top_k,
            sample_dockets=args.axisB_sample_dockets,
            concurrency=args.axisB_concurrency,
        )
        metrics.update({f'axisB_{k}': v for k, v in b.items()})
        logger.info('Axis B: %s', b)

    if 'axisC' in requested_steps:
        logger.info('=== Step 4: Axis C (frame distribution JSD) ===')
        c = await compute_axis_c(
            output_dir=output_dir,
            concurrency=args.axisC_concurrency,
        )
        metrics.update({f'axisC_{k}': v for k, v in c.items()})
        logger.info('Axis C: %s', c)

    if 'summary' in requested_steps:
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.info('Saved %s', metrics_path)


if __name__ == '__main__':
    asyncio.run(main())
