from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EvidenceType = Literal[
    "direct",
    "indirect",
    "proxy",
    "anecdotal",
    "contradictory",
    "irrelevant",
    "not_relevant",
]

SupportLevel = Literal[
    "strong",
    "moderate",
    "weak",
    "unsupported",
    "contradicted",
    "unknown",
]

SourceType = Literal[
    "paper",
    "benchmark",
    "blog_post",
    "company_claim",
    "case_study",
    "review",
    "dataset",
    "unknown",
]

QuestionStatus = Literal["open", "searched", "answered", "failed"]
ObservabilityBackend = Literal["local", "raindrop", "off"]
ObservabilityStatus = Literal["recorded", "skipped", "failed"]
ExecutionBackend = Literal["local", "modal"]
AgentOrchestrationMode = Literal["deterministic", "scripted_sdk", "live_sdk"]
AgentRunStatus = Literal["succeeded", "failed", "skipped"]
LiveRunMode = Literal["dry_run", "live"]
LiveRunStatus = Literal["ready", "succeeded", "blocked", "failed", "timed_out"]
ResearchTaskType = Literal[
    "retrieve_source",
    "parse_source",
    "extract_evidence",
    "cross_check",
    "verify_decisive_test",
]
ResearchTaskStatus = Literal["pending", "succeeded", "failed", "skipped"]
EvidenceConflictType = Literal[
    "contradiction",
    "source_type_imbalance",
    "weak_source_cluster",
    "duplicate_claim",
]
EvidenceConflictSeverity = Literal["low", "medium", "high"]
VerifierStatus = Literal["pass", "fail", "inconclusive"]
EvalWorkshopLinkType = Literal[
    "invalid_leap_to_eval",
    "evidence_conflict_to_invalid_leap",
    "verifier_failure_to_eval",
    "replay_eval_to_outcome",
]
RegressionEvalKind = Literal[
    "benchmark_proxy_boundary",
    "a6_requires_direct_validation",
    "company_claim_anecdotal",
    "conflict_workshop_links",
    "replay_confidence_not_increased",
]
EvalCaseStatus = Literal["pass", "fail"]
EvalSnapshotComparisonStatus = Literal["match", "changed", "regression"]
EvalFixtureDeltaStatus = Literal["same", "missing", "new", "changed"]


class Thesis(BaseModel):
    text: str
    domain: str | None = None


class Assumption(BaseModel):
    id: str
    text: str
    why_it_matters: str
    evidence_needed: list[str]
    support_level: SupportLevel = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latest_update: str | None = None


class ResearchQuestion(BaseModel):
    id: str
    assumption_ids: list[str]
    question: str
    query: str
    priority: int = Field(ge=1, le=5)
    status: QuestionStatus = "open"


class Source(BaseModel):
    id: str
    title: str
    source_type: SourceType
    url: str | None = None
    citation: str | None = None
    published_year: int | None = Field(default=None, ge=1900, le=2100)
    tags: list[str] = Field(default_factory=list)
    evidence_scope: str | None = None
    text: str


class RetrievalScore(BaseModel):
    id: str
    question_id: str
    source_id: str
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    rationale: str


class EvidenceItem(BaseModel):
    id: str
    source_id: str
    assumption_ids: list[str]
    evidence_type: EvidenceType
    claim_supported: str
    quoted_evidence: str
    limitation: str
    confidence: float = Field(ge=0.0, le=1.0)


class InvalidLeap(BaseModel):
    id: str
    leap: str
    why_invalid: str
    source_ids: list[str] = Field(default_factory=list)
    affected_assumption_ids: list[str]
    suggested_followup_question: str


class BeliefUpdate(BaseModel):
    assumption_id: str
    previous_support: SupportLevel
    new_support: SupportLevel
    previous_confidence: float = Field(ge=0.0, le=1.0)
    new_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class DecisiveTest(BaseModel):
    id: str
    test: str
    would_resolve: list[str]
    success_criteria: list[str]
    why_decisive: str


class GeneratedEval(BaseModel):
    id: str
    failure_observed: str
    root_cause: str
    eval_rule: str
    expected_behavior: str


class EvidenceConflict(BaseModel):
    id: str
    conflict_type: EvidenceConflictType
    severity: EvidenceConflictSeverity
    summary: str
    source_ids: list[str] = Field(default_factory=list)
    evidence_item_ids: list[str] = Field(default_factory=list)
    affected_assumption_ids: list[str]
    suggested_action: str


class VerifierResult(BaseModel):
    id: str
    decisive_test_id: str
    status: VerifierStatus
    affected_assumption_ids: list[str]
    confidence_delta: float = Field(ge=-1.0, le=1.0)
    rationale: str
    passed_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    evidence_item_ids: list[str] = Field(default_factory=list)


class ResearchTask(BaseModel):
    id: str
    task_type: ResearchTaskType
    question_ids: list[str] = Field(default_factory=list)
    assumption_ids: list[str] = Field(default_factory=list)
    source: Source | None = None
    sources: list[Source] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    decisive_test: DecisiveTest | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ResearchTaskResult(BaseModel):
    task_id: str
    task_type: ResearchTaskType
    backend: ExecutionBackend
    status: ResearchTaskStatus
    source_ids: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    verifier_results: list[VerifierResult] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class ResearchBatchResult(BaseModel):
    backend: ExecutionBackend
    attempted_backend: ExecutionBackend
    results: list[ResearchTaskResult] = Field(default_factory=list)
    fallback_reason: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    id: str
    stage: str
    message: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ObservabilityRecord(BaseModel):
    trace_id: str
    backend: ObservabilityBackend
    status: ObservabilityStatus
    trace_path: str | None = None
    workshop_path: str | None = None
    event_id: str | None = None
    eval_artifact_ids: list[str] = Field(default_factory=list)
    failure_artifact_ids: list[str] = Field(default_factory=list)
    workshop_artifact_ids: list[str] = Field(default_factory=list)
    message: str | None = None


class EvalWorkshopTaskSpan(BaseModel):
    id: str
    task_id: str
    task_type: ResearchTaskType
    backend: ExecutionBackend
    status: ResearchTaskStatus
    source_ids: list[str] = Field(default_factory=list)
    evidence_item_count: int = Field(ge=0)
    evidence_conflict_count: int = Field(ge=0)
    verifier_result_count: int = Field(ge=0)
    error: str | None = None


class EvalWorkshopLink(BaseModel):
    id: str
    link_type: EvalWorkshopLinkType
    source_id: str
    target_id: str
    summary: str
    affected_assumption_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class ReplayOutcomeRecord(BaseModel):
    id: str
    assumption_id: str
    generated_eval_id: str | None = None
    applied_eval_rule: str
    before_confidence: float = Field(ge=0.0, le=1.0)
    after_confidence: float = Field(ge=0.0, le=1.0)
    passed: bool
    summary: str


class EvalWorkshopRecord(BaseModel):
    task_spans: list[EvalWorkshopTaskSpan] = Field(default_factory=list)
    failure_eval_links: list[EvalWorkshopLink] = Field(default_factory=list)
    replay_outcomes: list[ReplayOutcomeRecord] = Field(default_factory=list)
    summary: str


class AgentRunStep(BaseModel):
    id: str
    tool_name: str
    status: AgentRunStatus
    summary: str
    agent_name: str | None = None


class AgentRunRecord(BaseModel):
    mode: AgentOrchestrationMode
    status: AgentRunStatus
    agent_name: str
    model: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    steps: list[AgentRunStep] = Field(default_factory=list)
    final_output_validated: bool = False
    message: str | None = None


class ResearchState(BaseModel):
    thesis: Thesis
    assumptions: list[Assumption] = Field(default_factory=list)
    research_questions: list[ResearchQuestion] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    retrieval_scores: list[RetrievalScore] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    invalid_leaps: list[InvalidLeap] = Field(default_factory=list)
    belief_updates: list[BeliefUpdate] = Field(default_factory=list)
    decisive_tests: list[DecisiveTest] = Field(default_factory=list)
    verifier_results: list[VerifierResult] = Field(default_factory=list)
    generated_evals: list[GeneratedEval] = Field(default_factory=list)
    research_task_results: list[ResearchTaskResult] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    observability: ObservabilityRecord | None = None
    eval_workshop: EvalWorkshopRecord | None = None
    agent_run: AgentRunRecord | None = None
    iteration: int = 0


class LiveRunGuardrails(BaseModel):
    mode: LiveRunMode = "dry_run"
    allow_live_sdk: bool = False
    prepared_corpus_only: bool = True
    allow_live_web_search: bool = False
    max_turns: int = Field(default=4, ge=1, le=20)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    max_iterations: int = Field(default=1, ge=1, le=5)
    execution_backend: ExecutionBackend = "local"
    observability_backend: ObservabilityBackend = "local"


class LiveRunResult(BaseModel):
    id: str
    created_at: str
    completed_at: str | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    mode: LiveRunMode
    status: LiveRunStatus
    thesis_text: str
    model: str | None = None
    guardrails: LiveRunGuardrails
    credentials_available: bool
    state: ResearchState | None = None
    output_path: str | None = None
    trace_path: str | None = None
    trace_id: str | None = None
    message: str
    error_type: str | None = None


class ReplayComparison(BaseModel):
    assumption_id: str
    before_support: SupportLevel
    after_support: SupportLevel
    before_confidence: float = Field(ge=0.0, le=1.0)
    after_confidence: float = Field(ge=0.0, le=1.0)
    change_summary: str
    rationale: str


class ReplayResult(BaseModel):
    first_pass: ResearchState
    replay_pass: ResearchState
    applied_eval_rules: list[str] = Field(default_factory=list)
    comparisons: list[ReplayComparison] = Field(default_factory=list)
    eval_workshop: EvalWorkshopRecord | None = None
    summary: str


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    thesis_text: str
    path: str
    trace_id: str | None = None
    assumption_count: int = Field(ge=0)
    evidence_item_count: int = Field(ge=0)
    evidence_conflict_count: int = Field(ge=0)
    invalid_leap_count: int = Field(ge=0)
    verifier_result_count: int = Field(ge=0)
    generated_eval_count: int = Field(ge=0)


class BeliefDelta(BaseModel):
    assumption_id: str
    assumption_text: str
    previous_support: SupportLevel
    current_support: SupportLevel
    previous_confidence: float = Field(ge=0.0, le=1.0)
    current_confidence: float = Field(ge=0.0, le=1.0)
    delta: float
    previous_update: str | None = None
    current_update: str | None = None


class RunComparison(BaseModel):
    baseline_run_id: str
    current_run_id: str
    deltas: list[BeliefDelta] = Field(default_factory=list)
    summary: str


class RegressionEvalCase(BaseModel):
    id: str
    name: str
    kind: RegressionEvalKind
    description: str
    expected_behavior: str


class RegressionEvalCaseResult(BaseModel):
    case: RegressionEvalCase
    status: EvalCaseStatus
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class RegressionEvalSuiteResult(BaseModel):
    id: str
    created_at: str
    status: EvalCaseStatus
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[RegressionEvalCaseResult] = Field(default_factory=list)
    summary: str


class GeneratedEvalFixture(BaseModel):
    id: str
    failure_observed: str
    root_cause: str
    eval_rule: str
    expected_behavior: str


class EvalSnapshotSummary(BaseModel):
    snapshot_id: str
    created_at: str
    thesis_text: str
    path: str
    suite_status: EvalCaseStatus
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    eval_case_count: int = Field(ge=0)
    generated_eval_count: int = Field(ge=0)


class EvalCorpusSnapshot(BaseModel):
    snapshot_id: str
    created_at: str
    thesis_text: str
    schema_version: str = "1"
    suite_result: RegressionEvalSuiteResult
    generated_eval_fixtures: list[GeneratedEvalFixture] = Field(default_factory=list)


class EvalCaseDelta(BaseModel):
    kind: RegressionEvalKind
    case_name: str
    baseline_status: EvalCaseStatus | None = None
    current_status: EvalCaseStatus | None = None
    changed: bool
    regression: bool
    baseline_message: str | None = None
    current_message: str | None = None


class GeneratedEvalFixtureDelta(BaseModel):
    eval_id: str
    status: EvalFixtureDeltaStatus
    baseline_eval_rule: str | None = None
    current_eval_rule: str | None = None
    baseline_expected_behavior: str | None = None
    current_expected_behavior: str | None = None


class EvalSnapshotComparison(BaseModel):
    baseline_snapshot_id: str
    current_snapshot_id: str
    status: EvalSnapshotComparisonStatus
    case_deltas: list[EvalCaseDelta] = Field(default_factory=list)
    fixture_deltas: list[GeneratedEvalFixtureDelta] = Field(default_factory=list)
    summary: str
