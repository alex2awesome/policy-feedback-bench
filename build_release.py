"""
build_release.py — build the PolicyFeedbackBench release artifacts.

Reads from the main regulations-demo data tree and writes trimmed,
shareable subsets into ./benchmark/ and ./reference_runs/. Run once
when the underlying data updates; re-running is idempotent.

Inputs (paths resolved relative to the regulations-demo root):
  data/bulk_downloads/deontic_units/comment_anticipation_benchmark_gpt5mini.jsonl
  data/bulk_downloads/deontic_units/proposed_rule_deontic_units_normalized_v2.jsonl
  data/bulk_downloads/deontic_units/match2_llm_labeled.jsonl
  data/feature_analysis_summary.csv.gz
  data/derived/paper_stats/stakeholder_frame_matrix.csv
  data/derived/real_claim_frames.jsonl                       (optional cache)
  data/derived/benchmark_runs/gpt-5-mini_vanilla/             (reference run)

Outputs (under ./benchmark/ and ./reference_runs/):
  benchmark/benchmark_dockets_558.jsonl                   # 558 docket IDs + n_units
  benchmark/proposed_rule_provisions_558.jsonl            # extracted deontic-unit provisions per docket
  benchmark/real_claims_matched_to_provisions_558.jsonl   # validated (real-claim, provision) pairs (Match-2 YES only)
  benchmark/commenter_org_type_lookup_558.csv.gz          # cluster_uid -> org_type/agency/year/primary_frame
  benchmark/stakeholder_frame_matrix.csv                  # 7-bucket frame distributions (paper Table 1)
  benchmark/real_claim_primary_frames_558.jsonl           # axis-C cache: real-claim primary_frame labels
  reference_runs/gpt-5-mini_vanilla/{generated_claims.jsonl, metrics.json}

Total release footprint after subsetting: ~60-80 MB.
"""
from __future__ import annotations
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REGDEMO = HERE.parent.parent  # data/policy-feedback-bench/ -> data/ -> regulations-demo/
DU = REGDEMO / 'data' / 'bulk_downloads' / 'deontic_units'
DERIVED = REGDEMO / 'data' / 'derived'

OUT_BENCH = HERE / 'benchmark'
OUT_RUNS = HERE / 'reference_runs'
OUT_BENCH.mkdir(parents=True, exist_ok=True)
OUT_RUNS.mkdir(parents=True, exist_ok=True)

# ---------------- step 1: load the 558 benchmark docket list ----------------
print('Loading benchmark docket list ...')
src_bench = DU / 'comment_anticipation_benchmark_gpt5mini.jsonl'
benchmark_dks = set()
docket_meta = []
seen = set()
with open(src_bench) as f:
    for line in f:
        try:
            r = json.loads(line)
            dk = r['docket_id']
            if dk in seen:
                continue
            seen.add(dk)
            benchmark_dks.add(dk)
            docket_meta.append({'docket_id': dk, 'n_units': r.get('n_units')})
        except Exception:
            pass
print(f'  {len(benchmark_dks):,} unique benchmark dockets')

with open(OUT_BENCH / 'benchmark_dockets_558.jsonl', 'w') as f:
    for r in docket_meta:
        f.write(json.dumps(r) + '\n')

# ---------------- step 2: subset proposed-rule units ----------------
print('Subsetting proposed-rule deontic units ...')
src_units = DU / 'proposed_rule_deontic_units_normalized_v2.jsonl'
n_in = n_out = 0
with open(src_units) as f, open(OUT_BENCH / 'proposed_rule_provisions_558.jsonl', 'w') as g:
    for line in f:
        n_in += 1
        try:
            r = json.loads(line)
            if r.get('docket_id') in benchmark_dks:
                g.write(line)
                n_out += 1
        except Exception:
            pass
print(f'  {n_out:,} / {n_in:,} doc-rows kept')

# ---------------- step 3: subset Match-2 YES pairs (gzipped: file is >100MB raw, exceeds GitHub limit) ----------------
print('Subsetting Match-2 YES pairs (filters to benchmark dockets + llm_label=1) ...')
src_match = DU / 'match2_llm_labeled.jsonl'
benchmark_cluster_uids = set()
n_in = n_out = 0
out_match = OUT_BENCH / 'real_claims_matched_to_provisions_558.jsonl.gz'
with open(src_match) as f, gzip.open(out_match, 'wt') as g:
    for line in f:
        n_in += 1
        if '"llm_label": 1' not in line:
            continue
        try:
            r = json.loads(line)
            if r.get('docket_id') in benchmark_dks and r.get('llm_label') == 1:
                # Keep only fields downstream needs; drop bi-encoder/CE scores (training-time
                # signals not used for evaluation) and n_comments (lookup-able elsewhere).
                slim = {k: r.get(k) for k in (
                    'docket_id', 'claim_id', 'cluster_uid', 'unit_id',
                    'unit_section', 'unit_type', 'unit_summary',
                    'claim_text', 'llm_label',
                ) if k in r}
                g.write(json.dumps(slim) + '\n')
                n_out += 1
                if r.get('cluster_uid'):
                    benchmark_cluster_uids.add(r['cluster_uid'])
        except Exception:
            pass
print(f'  {n_out:,} / {n_in:,} pairs kept; {len(benchmark_cluster_uids):,} unique cluster_uids')

# ---------------- step 4: subset cluster_uid -> org_type lookup ----------------
print('Subsetting cluster_uid -> org_type lookup ...')
src_feat = REGDEMO / 'data' / 'feature_analysis_summary.csv.gz'
df = pd.read_csv(src_feat, low_memory=False, dtype={'cluster_uid': str},
                  usecols=['cluster_uid', 'org_type', 'agency', 'year', 'primary_frame'])
df = df.dropna(subset=['cluster_uid'])
df = df[df['cluster_uid'].isin(benchmark_cluster_uids)]
df = df.drop_duplicates('cluster_uid', keep='first')
print(f'  {len(df):,} cluster_uid rows kept')
df.to_csv(OUT_BENCH / 'commenter_org_type_lookup_558.csv.gz', index=False, compression='gzip')

# ---------------- step 5: copy stakeholder_frame_matrix ----------------
print('Copying stakeholder_frame_matrix.csv ...')
src_fm = DERIVED / 'paper_stats' / 'stakeholder_frame_matrix.csv'
shutil.copy(src_fm, OUT_BENCH / 'stakeholder_frame_matrix.csv')

# ---------------- step 6: copy axisC real-claim-frame cache (if it exists) ----------------
src_rf = DERIVED / 'real_claim_frames.jsonl'
if src_rf.exists():
    print('Copying real_claim_primary_frames_558.jsonl (axisC cache) ...')
    shutil.copy(src_rf, OUT_BENCH / 'real_claim_primary_frames_558.jsonl')
else:
    print('Note: real_claim_frames.jsonl not found locally; consumers can regenerate it via run_benchmark.py at first run (~$4 in OpenAI cost).')

# ---------------- step 7: copy gpt-5-mini reference run ----------------
src_ref = DERIVED / 'benchmark_runs' / 'gpt-5-mini_vanilla'
dst_ref = OUT_RUNS / 'gpt-5-mini_vanilla'
dst_ref.mkdir(parents=True, exist_ok=True)
for fname in ('generated_claims.jsonl', 'metrics.json'):
    src = src_ref / fname
    if src.exists():
        print(f'Copying reference {src.name} ...')
        shutil.copy(src, dst_ref / fname)
    else:
        print(f'Note: reference {src} missing; reference run incomplete.')

print('\nDone. Release artifacts written to:')
for p in sorted(OUT_BENCH.glob('*')):
    sz = p.stat().st_size / 1024 / 1024
    print(f'  benchmark/{p.name:50s} {sz:6.1f} MB')
for p in sorted(OUT_RUNS.rglob('*')):
    if p.is_file():
        sz = p.stat().st_size / 1024 / 1024
        print(f'  reference_runs/{p.relative_to(OUT_RUNS)!s:50s} {sz:6.1f} MB')
