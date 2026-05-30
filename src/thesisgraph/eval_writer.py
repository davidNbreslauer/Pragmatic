from __future__ import annotations

from thesisgraph.schemas import GeneratedEval, InvalidLeap


def generate_evals_from_failures(invalid_leaps: list[InvalidLeap]) -> list[GeneratedEval]:
    evals: list[GeneratedEval] = []
    for index, leap in enumerate(invalid_leaps, start=1):
        evals.append(
            GeneratedEval(
                id=f"eval_{index:03d}",
                failure_observed=f"The system risked accepting this inference: {leap.leap}",
                root_cause=leap.why_invalid,
                eval_rule=_eval_rule_for_leap(leap),
                expected_behavior=(
                    "The agent should downgrade support, preserve the limitation, and ask for decisive follow-up evidence."
                ),
            )
        )
    return evals


def _eval_rule_for_leap(leap: InvalidLeap) -> str:
    if leap.id == "leap_benchmark_to_discovery":
        return (
            "If evidence only reports benchmark or QA performance, classify it as proxy evidence, "
            "not direct evidence for real-world discovery acceleration."
        )
    if leap.id == "leap_plausible_to_useful":
        return (
            "If evidence only reports plausible generated hypotheses, require novelty, testability, "
            "and validation before treating it as support for useful discovery."
        )
    if leap.id == "leap_graph_to_mechanism":
        return (
            "If evidence only shows graph representation or graph retrieval, do not infer causal "
            "or mechanistic understanding without ablations."
        )
    if leap.id == "leap_claim_to_validated_outcome":
        return (
            "If evidence is a company claim, classify it as anecdotal until independent validation is present."
        )
    return "Classify the evidence by what it directly measures, not by the stronger thesis it suggests."

