# PolicyFeedbackBench

A benchmark for evaluating whether large language models can anticipate the public comments that real stakeholders submit on proposed U.S. federal regulations.

## What this benchmark measures

For each of 558 federal rulemaking dockets (2016–2026), the benchmark gives the model the docket's proposed-rule **provisions** (extracted as atomic deontic units) and asks the model to generate the **comments** that real stakeholders would likely submit. We then score the model on four axes:

| Axis | What it measures | How |
|---|---|---|
| **A — Provision targeting** | Does the model comment on the *right* provisions? | Recall, Precision, and Pearson(ρ) of model claim counts vs. real claim counts per provision |
| **B — Claim semantic alignment** | Are the model's claims paraphrases of real-commenter claims? | LLM judge labels each (model, real) pair as `same / related / different / opposing`; reports `Strict` (`same` only) and `Loose` (`same`+`related`) match rates |
| **C — Frame distribution** | Does the model match the topical mix of real commenters? | Jensen–Shannon divergence of per-docket frame distributions over a 12-class taxonomy (health, safety, economic, legal, environmental, …) |
| **D — Substantivity** | Are the model's claims plausible enough that an agency would respond? | A separate LoRA-tuned Llama-3.1-8B classifier predicts the agency-response probability of each claim |

A bucket-alignment supplement also reports **whose stakeholder voice** the model implicitly takes (`industry / labor / advocacy / expert / government / law firm / individual`).

See the paper for full methodology and results across 8 model × 2 prompting variants.

## Quickstart

```bash
# 1. install dependencies
pip install openai pandas numpy scikit-learn sentence-transformers

# 2. set your API key (for OpenAI models)
export OPENAI_API_KEY=sk-...

# 3. run the benchmark on a model — generation + axes A, B, C
python run_benchmark.py \
    --model gpt-4o-mini \
    --provider openai \
    --variant vanilla \
    --output-name my_first_run \
    --steps gen,axisA,axisB,axisC,summary
```

For local open-weight models, run a [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) and pass `--provider vllm --base-url http://localhost:8001/v1 --model meta-llama/Llama-3.1-8B-Instruct`.

For the `subgroup` variant (recommended), pass `--variant subgroup` — the harness will run 7 stakeholder-persona generations per docket and pool the outputs.

Outputs land under `reference_runs/<output-name>/`:

```
reference_runs/my_first_run/
├── generated_claims.jsonl     # one row per docket with parsed claims
├── pair_judgments.jsonl       # axis B per-pair labels
├── llm_claim_frames.jsonl     # axis C per-claim primary_frame labels
└── metrics.json               # final aggregate metrics
```

## Cost-control flags

Axis B is the costliest step (LLM judge over many claim pairs). Two flags bound the cost:

- `--axisB-top-k 3` — for each real claim, judge against only the top-K most-similar model claims (default 3, set to 1 for cheapest)
- `--axisB-sample-dockets 100` — restrict axis B evaluation to 100 random dockets (rather than all 558); axis A and axis C still use the full set

A typical OpenAI-API run on the 8 models in the paper cost ~$25–35 total (axes A+B+C, with subgroup variant) at gpt-5-mini-flex pricing.

## Files

```
benchmark/
├── benchmark_dockets_558.jsonl                         # 558-docket index (docket_id, n_units)
├── proposed_rule_provisions_558.jsonl                  # extracted deontic-unit provisions per docket
├── real_claims_matched_to_provisions_558.jsonl         # validated (real-claim, provision) pairs
├── commenter_org_type_lookup_558.csv.gz                # cluster_uid -> org_type/agency/year/primary_frame
├── stakeholder_frame_matrix.csv                        # 7-bucket frame distributions (paper Table 1)
└── real_claim_primary_frames_558.jsonl                 # axis-C cache: primary_frame labels for the 43,371 real claims

reference_runs/
└── gpt-5-mini_vanilla/
    ├── generated_claims.jsonl                          # the canonical reference run
    └── metrics.json                                    # reference axis A/B/C scores

run_benchmark.py                                        # the harness (one-stop runner)
build_release.py                                        # rebuilds the data subsets from the source corpus
```

## Reproducing the paper's full results table

Each row of paper Table 2 corresponds to one `--model {...} --variant {vanilla,subgroup}` invocation. Reference outputs for the 8 model × 2 variant rows live under `reference_runs/<model>_<variant>/`.

To recompute Axis A on a finished run:

```bash
python run_benchmark.py --model <whatever> --provider openai \
    --output-name <existing-run> --steps axisA,summary
```

(The harness skips already-completed steps if their output files already exist.)

## Data provenance

- **Provisions** were extracted from federal-rule documents (2016–2026) by gpt-5-mini using a 6-shot prompt; full pipeline described in the paper appendix and at https://www.yalejreg.com/nc/only-one-third-of-proposed-regulatory-obligations-survive-to-the-final-rule/.
- **Real claims** were extracted from public comments and matched to provisions via a bi-encoder → ModernBERT cross-encoder → gpt-5-mini judge cascade with explicit opposing-sides framing (calibrated on the FCC RIF ground truth from Handan-Nader 2023, achieving 96% agreement with hand-curated labels).
- **Org_type** assignments come from the regulations.gov metadata field, mapped to a 7-bucket taxonomy.

## Citation

```bibtex
@inproceedings{spangher2026policyfeedbackbench,
  title     = {PolicyFeedbackBench: Evaluating LLM Anticipation of Federal Public Comments},
  author    = {Spangher, Alexander and Yang, Diyi and Koyejo, Sanmi and Ho, Daniel E.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2026}
}
```

(Bib key updated post-acceptance; please cite the paper PDF for now.)

## License

- **Code** (`run_benchmark.py`, `build_release.py`): MIT.
- **Data** (`benchmark/`, `reference_runs/`): CC-BY-4.0. The underlying federal-register text and public-comment text are U.S. government works in the public domain; LLM-derived annotations (Match-2 labels, frame labels, deontic-unit extractions) are released under CC-BY-4.0.

See `LICENSE` for full terms.
