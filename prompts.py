"""
prompts.py — every LLM prompt used by the benchmark, in one place.

Two categories of prompts:

1. **Generation prompts** (sent to the model under evaluation): ask the model
   to produce anticipated public-comment claims for a docket's provisions.
   Two variants:
     - VANILLA: free-form generation, no stakeholder conditioning.
     - SUBGROUP: same task, but conditioned on one of seven explicit
       stakeholder personas (industry, labor, advocacy NGO, expert
       organization, government agency, law firm, individual citizen).

2. **Evaluation prompts** (sent to a fixed judge model — gpt-5-mini by
   default): label generated outputs for the alignment and frame metrics.
     - PAIR_PROMPT: judges (model claim, real claim) pairs as
       same / related / different / opposing.
     - FRAME_PROMPT: assigns each claim a primary topical frame.

Editing prompt strings here changes benchmark behavior — proceed with care
if you want results comparable to the published paper.
"""
from __future__ import annotations

# ============================================================================
# Generation prompts (sent to the model under evaluation)
# ============================================================================

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


# Maps each stakeholder bucket key to a (short label, paragraph-long persona description).
# Used by GEN_PROMPT_SUBGROUP — the harness fires one prompt per persona per docket
# and pools the outputs.
STAKEHOLDER_PERSONAS = {
    'industry': (
        'industry / corporation / trade association',
        "a regulated company, trade association, or industry coalition. You focus on "
        "compliance cost, operational feasibility, market effects, and ROI. You cite "
        "economic data and industry-specific operational details.",
    ),
    'labor': (
        'labor union representative',
        "a labor union representing workers in the regulated industry. You focus on "
        "worker safety, job protection, fair labor standards, and economic impact on "
        "workers. You cite worker statistics and on-the-job experience.",
    ),
    'advocacy': (
        'public-interest advocacy NGO',
        "an environmental, civil rights, or public-interest advocacy organization. You "
        "focus on broader public good, environmental protection, equity, and the "
        "regulation's role in achieving social goals. You cite scientific studies, "
        "public-health data, and advocacy research.",
    ),
    'expert': (
        'expert / academic organization',
        "an academic researcher, professional society, or expert organization. You focus "
        "on technical correctness, scientific accuracy, peer-reviewed evidence, and "
        "methodological soundness. You cite research, technical specifications, and "
        "disciplinary expertise.",
    ),
    'gov': (
        'government / state agency',
        "a state or local government agency that interacts with this federal regulation. "
        "You focus on inter-jurisdictional coordination, implementation feasibility for "
        "state agencies, and federalism issues. You cite state-level data and "
        "operational details.",
    ),
    'lawfirm': (
        'law firm / bar association',
        "a law firm or bar association representing regulated parties. You focus on "
        "statutory authority, constitutional issues, APA procedural requirements, "
        "due-process implications, and legal interpretation. You cite case law, "
        "statutes, and regulatory history.",
    ),
    'individual': (
        'individual citizen or affected community member',
        "an ordinary individual citizen affected by this rule. You focus on personal "
        "experience, lived impact on you or your family/community, and concrete harms "
        "or benefits you would face. You speak in first person about direct experience.",
    ),
}


# ============================================================================
# Evaluation prompts (sent to the judge model)
# ============================================================================

# Includes the explicit "opposing-sides" framing that raised judge agreement
# with hand-curated ground truth from 67% to 96% in our calibration study
# (FCC RIF reference set, n=274). Without that framing the judge tends to
# label opposite-direction matches as "different" instead of "opposing".
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


# Default judge / labeler model. Override at the call site if you want a
# different judge for ablation studies.
JUDGE_MODEL = 'gpt-5-mini'
FRAME_MODEL = 'gpt-5-mini'


# Canonical 12-class frame taxonomy — must match the FRAME_PROMPT enum.
# Used by the JSD computation in metrics.py to keep distributions vector-aligned.
FRAMES = [
    'environmental', 'health', 'equity_justice', 'economic_cost_benefit',
    'legal_constitutional', 'religious_moral', 'safety', 'scientific_technical',
    'procedural', 'national_security', 'other', 'unknown',
]

# Map looser/older frame names that occasionally come back from the judge to
# canonical members of FRAMES. Anything unmapped falls through to 'unknown'.
FRAME_ALIASES = {
    'public_health': 'health',
    'moral_religious': 'religious_moral',
    'moral': 'religious_moral',
    'public_safety': 'safety',
    'consumer_protection': 'other',
    'consumer': 'other',
    'consumer_impact': 'other',
    'transparency_accountability': 'procedural',
    'animal_welfare': 'environmental',
    'human_rights': 'equity_justice',
    'individual_rights': 'equity_justice',
    'humanitarian': 'equity_justice',
    'cultural': 'other',
    'technical': 'scientific_technical',
    'professional_society': 'other',
    'privacy': 'legal_constitutional',
    'educational': 'other',
}
