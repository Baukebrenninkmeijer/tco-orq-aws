"""Cost calculation functions for AWS and Orq TCO comparison."""

from dataclasses import dataclass

from config import AWSPricing, OrqPricing, RuntimeParams, ScenarioParams


@dataclass
class CostBreakdown:
    """Breakdown of costs by component."""

    base: float = 0.0
    runtime: float = 0.0
    gateway: float = 0.0
    memory: float = 0.0
    observability: float = 0.0
    evaluations: float = 0.0

    @property
    def total(self) -> float:
        """Total cost across all components."""
        return self.base + self.runtime + self.gateway + self.memory + self.observability + self.evaluations


def calculate_runtime_hours(runtime: RuntimeParams) -> tuple[float, float]:
    """
    Calculate CPU-hours and GB-hours per trace.

    Args:
        runtime: Runtime parameters (session duration, CPU cores, memory GB)

    Returns:
        Tuple of (cpu_hours_per_trace, gb_hours_per_trace)
    """
    hours_per_trace = runtime.session_duration_seconds / 3600
    cpu_hours = hours_per_trace * runtime.cpu_cores
    gb_hours = hours_per_trace * runtime.memory_gb
    return cpu_hours, gb_hours


def calculate_aws_costs(
    traces: int,
    scenario: ScenarioParams,
    runtime: RuntimeParams,
    pricing: AWSPricing,
    eur_usd_rate: float,
) -> CostBreakdown:
    """
    Calculate AWS costs for a given number of traces.

    All costs are calculated in USD and converted to EUR.

    Args:
        traces: Number of traces
        scenario: Scenario parameters
        runtime: Runtime parameters
        pricing: AWS pricing configuration
        eur_usd_rate: EUR to USD exchange rate

    Returns:
        CostBreakdown with all cost components in EUR
    """
    cpu_hours, gb_hours = calculate_runtime_hours(runtime)
    gb_per_trace = scenario.gb_per_trace

    # Base cost (AWS has no base subscription)
    base_usd = 0.0

    # Runtime cost
    runtime_usd = traces * (cpu_hours * pricing.cpu_hour + gb_hours * pricing.gb_hour)

    # Gateway cost (search + tool calls + tools indexed)
    gateway_usd = (
        (scenario.search_calls_per_trace * traces / 1e6 * pricing.search_tool_per_m)
        + (scenario.tool_calls_per_trace * traces / 1e6 * pricing.tool_call_per_m)
        + (scenario.tools_indexed / 100 * pricing.tools_indexed_per_100)  # Monthly cost
    )

    # Memory cost (short-term events + long-term storage + retrievals)
    memory_usd = (
        (scenario.short_term_memory_events_per_trace * traces / 1000 * pricing.memory_short_term_per_k)
        + (scenario.long_term_memories_per_trace * traces / 1000 * pricing.memory_long_term_per_k)
        + (scenario.memory_retrievals_per_trace * traces / 1000 * pricing.memory_retrieval_per_k)
    )

    # Observability cost (span ingestion + event logging)
    # Event logging is estimated at 60% of span data
    observability_usd = (traces * gb_per_trace * pricing.span_ingestion_per_gb) + (
        0.6 * traces * gb_per_trace * pricing.event_logging_per_gb
    )

    # Evaluation costs
    # Built-in evals (input + output tokens)
    builtin_evals_input_usd = (
        scenario.total_evaluations
        * scenario.tokens_per_trace
        * scenario.builtin_evals_count
        * pricing.builtin_evals_input_per_m_tokens
        / 1e6
    )
    builtin_evals_output_usd = (
        scenario.total_evaluations
        * scenario.eval_output_tokens
        * scenario.builtin_evals_count
        * pricing.builtin_evals_output_per_m_tokens
        / 1e6
    )
    # Custom evals
    custom_evals_usd = (
        scenario.total_evaluations * scenario.custom_evals_count * pricing.custom_evals_per_k_calls / 1000
    )
    evaluations_usd = builtin_evals_input_usd + builtin_evals_output_usd + custom_evals_usd

    # Convert all to EUR
    return CostBreakdown(
        base=base_usd / eur_usd_rate,
        runtime=runtime_usd / eur_usd_rate,
        gateway=gateway_usd / eur_usd_rate,
        memory=memory_usd / eur_usd_rate,
        observability=observability_usd / eur_usd_rate,
        evaluations=evaluations_usd / eur_usd_rate,
    )


def calculate_orq_costs(
    traces: int,
    scenario: ScenarioParams,
    runtime: RuntimeParams,
    pricing: OrqPricing,
) -> CostBreakdown:
    """
    Calculate Orq costs for a given number of traces.

    All costs are in EUR.

    Args:
        traces: Number of traces
        scenario: Scenario parameters
        runtime: Runtime parameters
        pricing: Orq pricing configuration

    Returns:
        CostBreakdown with all cost components in EUR
    """
    gb_per_trace = scenario.gb_per_trace

    # Base subscription
    base_eur = pricing.base_subscription

    # Runtime cost (per second pricing)
    runtime_eur = traces * runtime.session_duration_seconds * pricing.cost_per_second

    # Gateway cost (included in Orq)
    gateway_eur = 0.0

    # Memory cost (included in Orq base subscription)
    memory_eur = 0.0

    # Calculate total storage used (traces + evaluations)
    eval_traces_count = scenario.total_evaluations * (scenario.builtin_evals_count + scenario.custom_evals_count)
    eval_storage = eval_traces_count * gb_per_trace
    trace_storage = traces * gb_per_trace
    total_storage_used = trace_storage + eval_storage

    # Observability cost (traces overage + storage overage)
    traces_overage = max(traces - pricing.included_traces, 0)
    traces_overage_cost = traces_overage * pricing.traces_overage_per_k / 1000

    storage_overage = max(total_storage_used - pricing.included_storage_gb, 0)
    storage_overage_cost = storage_overage * pricing.storage_per_gb

    observability_eur = traces_overage_cost + storage_overage_cost

    # Evaluation costs for Orq: included in storage calculation above, no separate charge
    evaluations_eur = 0.0

    return CostBreakdown(
        base=base_eur,
        runtime=runtime_eur,
        gateway=gateway_eur,
        memory=memory_eur,
        observability=observability_eur,
        evaluations=evaluations_eur,
    )


def calculate_costs_for_trace_range(
    trace_points: list[int],
    scenario: ScenarioParams,
    runtime: RuntimeParams,
    aws_pricing: AWSPricing,
    orq_pricing: OrqPricing,
    eur_usd_rate: float,
) -> tuple[list[CostBreakdown], list[CostBreakdown]]:
    """
    Calculate costs for a range of trace counts.

    Args:
        trace_points: List of trace counts to calculate
        scenario: Scenario parameters
        runtime: Runtime parameters
        aws_pricing: AWS pricing configuration
        orq_pricing: Orq pricing configuration
        eur_usd_rate: EUR to USD exchange rate

    Returns:
        Tuple of (aws_costs, orq_costs) where each is a list of CostBreakdown
    """
    aws_costs = []
    orq_costs = []

    for traces in trace_points:
        aws_costs.append(
            calculate_aws_costs(traces, scenario, runtime, aws_pricing, eur_usd_rate)
        )
        orq_costs.append(
            calculate_orq_costs(traces, scenario, runtime, orq_pricing)
        )

    return aws_costs, orq_costs
