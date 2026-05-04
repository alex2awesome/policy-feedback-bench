"""
paths.py — central file-path configuration for PolicyFeedbackBench.

All other modules import paths from here so the layout is in one place.
The package root is auto-detected from this file's location, but you can
override it by setting POLICY_FEEDBACK_BENCH_ROOT in your environment
(useful when running from a different working directory).
"""
from __future__ import annotations
import os
from pathlib import Path

# Package root: the directory containing this file.
# Override via environment variable if you are running from elsewhere.
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('POLICY_FEEDBACK_BENCH_ROOT', HERE))

# Top-level subdirectories.
BENCHMARK_DIR = ROOT / 'benchmark'         # shipped input data (one-time prepared by build_release.py)
RUNS_DIR = ROOT / 'reference_runs'         # per-model output directories live here

# Individual benchmark input files (consumers do not modify these).
DOCKETS_FILE = BENCHMARK_DIR / 'benchmark_dockets_558.jsonl'
PROVISIONS_FILE = BENCHMARK_DIR / 'proposed_rule_provisions_558.jsonl'
REAL_CLAIMS_FILE = BENCHMARK_DIR / 'real_claims_matched_to_provisions_558.jsonl.gz'
ORG_TYPE_LOOKUP_FILE = BENCHMARK_DIR / 'commenter_org_type_lookup_558.csv.gz'
STAKEHOLDER_FRAME_MATRIX_FILE = BENCHMARK_DIR / 'stakeholder_frame_matrix.csv'
REAL_CLAIM_FRAMES_FILE = BENCHMARK_DIR / 'real_claim_primary_frames_558.jsonl'

# Per-run output filenames (created under RUNS_DIR/<output_name>/).
GENERATED_CLAIMS_FILENAME = 'generated_claims.jsonl'
PAIR_JUDGMENTS_FILENAME = 'pair_judgments.jsonl'
LLM_CLAIM_FRAMES_FILENAME = 'llm_claim_frames.jsonl'
METRICS_FILENAME = 'metrics.json'
