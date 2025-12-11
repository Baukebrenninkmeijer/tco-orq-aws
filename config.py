"""Configuration defaults and presets for TCO comparison app."""

from dataclasses import dataclass, field


@dataclass
class AWSPricing:
    """AWS pricing parameters (all in USD)."""

    # Runtime (per vCPU-hour / GB-hour)
    cpu_hour: float = 0.0895
    gb_hour: float = 0.00945

    # Gateway
    search_tool_per_m: float = 25.0  # $0.025 per 1k = $25 per M
    tool_call_per_m: float = 5.0  # $0.005 per 1k = $5 per M
    tools_indexed_per_100: float = 0.02  # $0.02 per 100 tools indexed/month

    # Memory
    memory_short_term_per_k: float = 0.25  # $0.25 per 1k short-term events
    memory_long_term_per_k: float = 0.75  # $0.75 per 1k long-term memories stored
    memory_retrieval_per_k: float = 0.50  # $0.50 per 1k retrievals

    # Observability (CloudWatch)
    span_ingestion_per_gb: float = 0.35
    event_logging_per_gb: float = 0.50

    # Evaluations
    builtin_evals_input_per_m_tokens: float = 2.40
    builtin_evals_output_per_m_tokens: float = 12.0
    custom_evals_per_k_calls: float = 1.50


@dataclass
class OrqPricing:
    """Orq pricing parameters (all in EUR)."""

    base_subscription: float = 250.0
    included_traces: int = 1_000_000
    included_storage_gb: float = 50.0
    traces_overage_per_k: float = 1.50
    storage_per_gb: float = 3.0
    # Runtime costs
    cost_per_second: float = 0.0


@dataclass
class RuntimeParams:
    """Runtime parameters for compute cost calculation."""

    session_duration_seconds: float = 10.0
    cpu_cores: float = 0.25
    memory_gb: float = 0.5


@dataclass
class ScenarioParams:
    """Parameters defining a usage scenario."""

    name: str = "Custom"
    spans_per_trace: int = 15
    search_calls_per_trace: int = 1
    tool_calls_per_trace: int = 3
    tokens_per_trace: int = 15_000
    bytes_per_token: int = 4
    total_evaluations: int = 15_000
    builtin_evals_count: int = 0
    custom_evals_count: int = 5
    eval_output_tokens: int = 300

    # Memory parameters (per trace)
    short_term_memory_events_per_trace: int = 1  # Events created per trace
    long_term_memories_per_trace: float = 0.5  # Long-term memories stored per trace
    memory_retrievals_per_trace: int = 2  # Memory retrievals per trace

    # Gateway parameters
    tools_indexed: int = 10  # Number of tools indexed in Gateway

    @property
    def gb_per_trace(self) -> float:
        """Calculate GB per trace from tokens and bytes per token."""
        return (self.bytes_per_token * self.tokens_per_trace) / 1_000_000_000


# Preset scenarios
PRESETS: dict[str, ScenarioParams] = {
    "Support Bot Large": ScenarioParams(
        name="Support Bot Large",
        spans_per_trace=15,
        search_calls_per_trace=1,
        tool_calls_per_trace=3,
        tokens_per_trace=15_000,
        bytes_per_token=4,
        total_evaluations=15_000,
        builtin_evals_count=0,
        custom_evals_count=5,
        eval_output_tokens=300,
        short_term_memory_events_per_trace=4,
        long_term_memories_per_trace=0.5,
        memory_retrievals_per_trace=4,
        tools_indexed=30,
    ),
    "Support Bot Small": ScenarioParams(
        name="Support Bot Small",
        spans_per_trace=15,
        search_calls_per_trace=2,
        tool_calls_per_trace=8,
        tokens_per_trace=15_000,
        bytes_per_token=4,
        total_evaluations=15_000,
        builtin_evals_count=0,
        custom_evals_count=5,
        eval_output_tokens=300,
        short_term_memory_events_per_trace=2,
        long_term_memories_per_trace=0.2,
        memory_retrievals_per_trace=1,
        tools_indexed=10,
    ),
    "Multi-agent": ScenarioParams(
        name="Multi-agent",
        spans_per_trace=15,
        search_calls_per_trace=2,
        tool_calls_per_trace=8,
        tokens_per_trace=80_000,
        bytes_per_token=4,
        total_evaluations=15_000,
        builtin_evals_count=0,
        custom_evals_count=5,
        eval_output_tokens=300,
        short_term_memory_events_per_trace=3,
        long_term_memories_per_trace=1.0,
        memory_retrievals_per_trace=6,
        tools_indexed=100,
    ),
    "Custom": ScenarioParams(name="Custom"),
}

# Currency conversion
DEFAULT_EUR_USD_RATE = 1.17

# Trace range for visualization
TRACE_POINTS = [
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
]
