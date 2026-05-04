# PolicyFeedbackBench

A benchmark for evaluating whether large language models can anticipate the public comments that real stakeholders submit on proposed U.S. federal regulations.

## What this benchmark measures

For each of 558 federal rulemaking dockets (2016–2026), the benchmark gives the model the docket's proposed-rule **provisions** (extracted as atomic deontic units) and asks the model to generate the **comments** that real stakeholders would likely submit. We then score the model on four axes (described in detail in [§ Metrics](#metrics) below):

| Axis | What it measures | Key metric(s) |
|---|---|---|
| **A — Provision targeting** | Does the model comment on the *right* provisions? | Recall, Precision, Pearson(ρ) |
| **B — Claim semantic alignment** | Are the model's claims paraphrases of real-commenter claims? | Strict, Loose |
| **C — Frame distribution** | Does the model match the topical mix of real commenters? | JSD |
| **D — Substantivity** | Are the model's claims plausible enough that an agency would respond? | AUC |

A bucket-alignment supplement also reports **whose stakeholder voice** the model implicitly takes (`industry / labor / advocacy / expert / government / law firm / individual`).

## Quickstart

```bash
# 1. Install dependencies
pip install openai anthropic pandas numpy scikit-learn sentence-transformers

# 2. Set your API key
export OPENAI_API_KEY=sk-...
# (or ANTHROPIC_API_KEY=... if calling claude models)

# 3. Run the full pipeline on a model
python run_benchmark.py \
    --model gpt-4o-mini \
    --provider openai \
    --variant vanilla \
    --output-name gpt-4o-mini_vanilla \
    --steps gen,axisA,axisB,axisC,summary
```

This will (1) generate predicted comments per docket, (2) score axis A, (3) score axis B (with cost-control defaults; see flags below), (4) score axis C, and (5) write a final `metrics.json`.

### Running on a local open-weight model
Stand up a [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) and pass:

```bash
python run_benchmark.py \
    --provider vllm \
    --base-url http://localhost:8001/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --variant vanilla \
    --output-name llama-3.1-8b_vanilla \
    --steps gen,axisA,axisB,axisC,summary
```

### Recommended: subgroup variant
Pass `--variant subgroup` and the harness runs **7 stakeholder-persona generations per docket** (industry, labor, advocacy NGO, expert org, government agency, law firm, individual citizen) and pools the outputs. In our paper this roughly **doubles** strict claim-alignment for every model tested.

### Cost-control flags
Axis B is the costliest step (LLM judge over many claim pairs). Two flags bound the cost:

| Flag | Effect |
|---|---|
| `--axisB-top-k 3` | For each real claim, judge against only the top-K most-similar model claims (default 3; set to 1 for cheapest) |
| `--axisB-sample-dockets 100` | Restrict axis B evaluation to 100 random dockets (axes A and C still use the full 558) |

A typical OpenAI-API run on one model (axes A+B+C, subgroup variant, full 558 dockets, K=3) costs **~$2–5** at gpt-5-mini-flex pricing. If you only want axis A (cheapest, no LLM judging), use `--steps gen,axisA,summary` and total cost is ~$0.50 per model.

### Re-running individual steps
The harness writes a separate output file per step and skips work it has already done (resume support):

```bash
# Just recompute axis A on an existing run
python run_benchmark.py --model gpt-4o-mini --provider openai \
    --output-name gpt-4o-mini_vanilla \
    --steps axisA,summary
```

If you want to start fresh, delete the run directory: `rm -rf reference_runs/<output-name>/`.

---

## Metrics

All metrics are computed against the population of validated **real-commenter claims** in `benchmark/real_claims_matched_to_provisions_558.jsonl.gz`. Each docket has between a handful and several hundred real claims; total is **43,371 unique real claims** across all 558 dockets.

### Axis A — Provision targeting
Does the model comment on the same provisions as real commenters, with the same intensity?

- **Recall** = per-docket fraction of real-comment-targeted provisions that the model also targets (i.e., generates ≥ 1 claim against). Bounded [0, 1]; higher is better.
- **Precision** = per-docket fraction of model-targeted provisions that real commenters also targeted. Penalizes models that "spam" claims on uncommented provisions. Bounded [0, 1]; higher is better.
- **Pearson(ρ)** = correlation between (i) per-provision count of real comment clusters and (ii) per-provision count of model claims, pooled across all dockets. A high ρ means the model not only hits the right provisions but also calibrates the *intensity* of attention to match the heavy-tailed real distribution. Bounded [-1, 1]; higher is better.

### Axis B — Claim semantic alignment
For each real claim, do the model's claims actually paraphrase it?

We use **gpt-5-mini as a judge** (with explicit opposing-sides framing — see paper §2.4 for calibration). For each (real claim, candidate model claim) pair on the same provision, the judge labels:

| Label | Meaning |
|---|---|
| `same` | model claim is a paraphrase / restatement of the real claim, same direction |
| `related` | same sub-topic, different point, compatible direction |
| `different` | addresses a different aspect of the same provision |
| `opposing` | takes a contrary position on the same point |

We then report:
- **Strict** = % of real claims with at least one `same` model match
- **Loose** = % of real claims with at least one `same` *or* `related` model match

A real claim with no model claim on its provision counts against both. By default we judge each real claim against the **top-3 most-similar** model claims (cosine on `all-MiniLM-L6-v2` embeddings) — see `--axisB-top-k`.

### Axis C — Frame distribution faithfulness
Does the model match the *topical mix* of real commenters?

- We label every claim (real and model-generated) with a **12-class primary-frame** taxonomy: `health, safety, economic_cost_benefit, legal_constitutional, scientific_technical, environmental, equity_justice, religious_moral, national_security, procedural, other, unknown`.
- Per docket, we compute the frame distribution over real claims and over model claims.
- **JSD** = Jensen–Shannon divergence between those two distributions, in `[0, ln 2]`. Lower is better; 0 is a perfect match.
- We report mean and median JSD across the 558 dockets.

The real-claim frame labels are **shipped pre-computed** in `benchmark/real_claim_primary_frames_558.jsonl` (so consumers don't re-pay for the ~$4 in OpenAI cost). Model-claim frame labels are computed per run and cached in `reference_runs/<output-name>/llm_claim_frames.jsonl`.

### Axis D — Substantivity
Are the model's claims plausible enough that an agency would respond?

A separate **LoRA-tuned Llama-3.1-8B classifier** (trained on ~13K real (claim, was-the-claim-responded-to) labels) predicts the agency-response probability of each claim. We report **ROC-AUC** of the classifier discriminating real claims from model claims on the same dockets. A model that is indistinguishable from real commenters scores AUC ≈ 0.5; a model whose claims are systematically less substantive scores higher.

> **Note:** Axis D requires running the classifier on a GPU (the Llama-3.1-8B LoRA adapter). It is **not** included in the default `run_benchmark.py` pipeline. We ship reference Axis D scores per model in the paper; reproducibility instructions are in [§ Axis D substantivity reproducibility](#axis-d-substantivity-reproducibility) below.

### Bucket alignment (supplement)
For each model we also compute *whose stakeholder voice* it implicitly adopts. For each real-commenter org type X, we compute `strict_align(X) = same / (same + opposing)` over all (model claim, real X claim) pair judgments. The most-aligned org is the model's "implicit voice." See `benchmark/commenter_org_type_lookup_558.csv.gz` for the org_type assignments.

---

## Where the metric files live

After running `run_benchmark.py --output-name <run-name>`, the following files are written under `reference_runs/<run-name>/`:

| File | Produced by step | Schema | Used in |
|---|---|---|---|
| `generated_claims.jsonl` | `gen` | one row per docket, fields `{docket_id, n_units, parsed_claims, variant, model}`; each parsed claim has `{provision_idx, claim_text, stakeholder}` | All downstream axes |
| `pair_judgments.jsonl` | `axisB` | one row per (real claim, model claim) pair, fields `{claim_id, llm_text_hash, label}` where label ∈ `{same, related, different, opposing, error}` | Axis B Strict/Loose |
| `llm_claim_frames.jsonl` | `axisC` | one row per model claim, fields `{claim_key, docket_id, frame}` (frame is one of the 12-class taxonomy values) | Axis C JSD |
| `metrics.json` | `summary` | aggregate metrics dict (see schema below) | Final reporting |

`metrics.json` schema:

```json
{
  "axisA_recall":               <float, mean across dockets>,
  "axisA_precision":            <float>,
  "axisA_f1":                   <float>,
  "axisA_pearson_attention":    <float>,
  "axisA_n_dockets_eval":       <int>,
  "axisB_strict":               <float, fraction of real claims with at least one 'same' match>,
  "axisB_loose":                <float, fraction with 'same' or 'related'>,
  "axisB_n_real_claims_evaluated":     <int, unique real claim_ids judged>,
  "axisB_n_real_claims_with_candidates": <int, claim_ids with at least 1 candidate model claim>,
  "axisB_n_pair_judgments":     <int, total pairs sent to the judge>,
  "axisB_tokens_in":            <int, judge input tokens>,
  "axisB_tokens_out":           <int, judge output tokens>,
  "axisC_jsd_mean":             <float, mean per-docket JSD>,
  "axisC_jsd_median":           <float>,
  "axisC_n_dockets_eval":       <int, dockets with both real + model frame labels>,
  "axisC_n_llm_claims_labeled": <int>,
  "axisC_n_real_claims_labeled":<int>
}
```

The reference run for `gpt-5-mini` (vanilla prompting) lives in `reference_runs/gpt-5-mini_vanilla/` so you can sanity-check your install by comparing `metrics.json`.

---

## Files

```
benchmark/
├── benchmark_dockets_558.jsonl                         # 558-docket index (docket_id, n_units)
├── proposed_rule_provisions_558.jsonl                  # extracted deontic-unit provisions per docket
├── real_claims_matched_to_provisions_558.jsonl.gz      # validated (real-claim, provision) pairs
├── commenter_org_type_lookup_558.csv.gz                # cluster_uid -> org_type/agency/year/primary_frame
├── stakeholder_frame_matrix.csv                        # 7-bucket frame distributions (paper Table 1)
└── real_claim_primary_frames_558.jsonl                 # axis-C cache: primary_frame for the 43,371 real claims

reference_runs/
└── gpt-5-mini_vanilla/                                 # canonical reference run (sanity-check target)
    ├── generated_claims.jsonl
    └── metrics.json

run_benchmark.py                                        # the harness (one-stop runner)
build_release.py                                        # rebuilds the data subsets from the source corpus
```

---

## Reproducing the paper's full results table
Each row of paper Table 2 corresponds to one `--model {...} --variant {vanilla,subgroup}` invocation. To run all 8 main rows (4 OpenAI + 2 Llama, vanilla and subgroup):

```bash
for variant in vanilla subgroup; do
  for model in gpt-4o-mini gpt-5-mini gpt-5; do
    python run_benchmark.py --model $model --provider openai --variant $variant \
        --output-name ${model}_${variant} --steps gen,axisA,axisB,axisC,summary
  done
  # Llama via local vLLM at port 8001
  python run_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct \
      --provider vllm --base-url http://localhost:8001/v1 --variant $variant \
      --output-name llama-3.1-8b-instruct_$variant --steps gen,axisA,axisB,axisC,summary
done
```

---

## Axis D substantivity reproducibility
Axis D requires the LoRA-tuned Llama-3.1-8B classifier (~16 GiB of GPU memory). Setup:

```bash
# 1. Download the trained adapter from <huggingface link TBD>
# 2. Run the classifier on each model's generated_claims.jsonl
python predict_substantivity.py \
    --adapter <path-to-lora> \
    --input reference_runs/<output-name>/generated_claims.jsonl \
    --output reference_runs/<output-name>/substantivity_preds.csv
# 3. Compute discriminator AUC
python compute_axisD.py \
    --real benchmark/real_claim_primary_frames_558.jsonl \
    --model reference_runs/<output-name>/substantivity_preds.csv
```

(`predict_substantivity.py` and `compute_axisD.py` will be added in a follow-up release.)

---

## Data provenance

- **Provisions** were extracted from federal-rule documents (2016–2026) by gpt-5-mini using a 6-shot prompt; full pipeline described in the paper appendix and at https://www.yalejreg.com/nc/only-one-third-of-proposed-regulatory-obligations-survive-to-the-final-rule/.
- **Real claims** were extracted from public comments and matched to provisions via a bi-encoder → ModernBERT cross-encoder → gpt-5-mini judge cascade with explicit opposing-sides framing (calibrated on the FCC RIF ground truth from Handan-Nader 2023, achieving 96% agreement with hand-curated labels).
- **Org_type** assignments come from the regulations.gov metadata field, mapped to a 7-bucket taxonomy.

---

## License

- **Code** (`run_benchmark.py`, `build_release.py`): MIT.
- **Data** (`benchmark/`, `reference_runs/`): CC-BY-4.0. The underlying federal-register text and public-comment text are U.S. government works in the public domain; LLM-derived annotations (claim-to-provision matches, frame labels, deontic-unit extractions) are released under CC-BY-4.0.

See `LICENSE` for full terms.
