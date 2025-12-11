"""Streamlit app for TCO comparison between AWS and Orq."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from calculations import (
    CostBreakdown,
    calculate_aws_costs,
    calculate_costs_for_trace_range,
    calculate_orq_costs,
)
from config import (
    DEFAULT_EUR_USD_RATE,
    PRESETS,
    TRACE_POINTS,
    AWSPricing,
    OrqPricing,
    RuntimeParams,
    ScenarioParams,
)

st.set_page_config(
    page_title="TCO Comparison: AWS vs Orq",
    page_icon="📊",
    layout="wide",
)

st.title("TCO Comparison: AWS vs Orq")
st.markdown("Compare total cost of ownership for agent-based AI systems")

# -----------------------------------------------------------------------------
# Sidebar: Pricing Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Pricing Configuration")

    # AWS Pricing
    with st.expander("AWS Pricing (USD)", expanded=False):
        st.markdown("**Runtime**")
        aws_cpu_hour = st.number_input(
            "CPU Hour", value=0.0895, format="%.4f", key="aws_cpu"
        )
        aws_gb_hour = st.number_input(
            "GB Hour", value=0.00945, format="%.5f", key="aws_gb"
        )
        st.markdown("**Gateway**")
        aws_search_per_m = st.number_input(
            "Search Tool (per M)", value=25.0, format="%.2f", key="aws_search"
        )
        aws_tool_per_m = st.number_input(
            "Tool Call (per M)", value=5.0, format="%.2f", key="aws_tool"
        )
        aws_tools_indexed = st.number_input(
            "Tools Indexed (per 100/month)", value=0.02, format="%.2f", key="aws_tools_idx"
        )
        st.markdown("**Memory**")
        aws_memory_short_term = st.number_input(
            "Short-term Events (per 1k)", value=0.25, format="%.2f", key="aws_mem_st"
        )
        aws_memory_long_term = st.number_input(
            "Long-term Memories (per 1k)", value=0.75, format="%.2f", key="aws_mem_lt"
        )
        aws_memory_retrieval = st.number_input(
            "Memory Retrievals (per 1k)", value=0.50, format="%.2f", key="aws_mem_ret"
        )
        st.markdown("**Observability**")
        aws_span_ingestion = st.number_input(
            "Span Ingestion (per GB)", value=0.35, format="%.2f", key="aws_span"
        )
        aws_event_logging = st.number_input(
            "Event Logging (per GB)", value=0.50, format="%.2f", key="aws_event"
        )
        st.markdown("**Evaluations**")
        aws_builtin_input = st.number_input(
            "Built-in Evals Input (per M tokens)",
            value=2.40,
            format="%.2f",
            key="aws_eval_in",
        )
        aws_builtin_output = st.number_input(
            "Built-in Evals Output (per M tokens)",
            value=12.0,
            format="%.2f",
            key="aws_eval_out",
        )
        aws_custom_evals = st.number_input(
            "Custom Evals (per 1k calls)",
            value=1.50,
            format="%.2f",
            key="aws_custom",
        )

    aws_pricing = AWSPricing(
        cpu_hour=aws_cpu_hour,
        gb_hour=aws_gb_hour,
        search_tool_per_m=aws_search_per_m,
        tool_call_per_m=aws_tool_per_m,
        tools_indexed_per_100=aws_tools_indexed,
        memory_short_term_per_k=aws_memory_short_term,
        memory_long_term_per_k=aws_memory_long_term,
        memory_retrieval_per_k=aws_memory_retrieval,
        span_ingestion_per_gb=aws_span_ingestion,
        event_logging_per_gb=aws_event_logging,
        builtin_evals_input_per_m_tokens=aws_builtin_input,
        builtin_evals_output_per_m_tokens=aws_builtin_output,
        custom_evals_per_k_calls=aws_custom_evals,
    )

    # Orq Pricing
    with st.expander("Orq Pricing (EUR)", expanded=False):
        orq_base = st.number_input(
            "Base Subscription", value=750.0, format="%.2f", key="orq_base"
        )
        orq_included_traces = st.number_input(
            "Included Traces", value=1_000_000, step=100_000, key="orq_traces"
        )
        orq_included_storage = st.number_input(
            "Included Storage (GB)", value=50.0, format="%.1f", key="orq_storage"
        )
        orq_trace_overage = st.number_input(
            "Traces Overage (per 1k)", value=1.50, format="%.2f", key="orq_trace_over"
        )
        orq_storage_price = st.number_input(
            "Storage (per GB)", value=3.0, format="%.2f", key="orq_storage_price"
        )
        st.markdown("---")
        st.markdown("*Runtime (placeholder for future)*")
        orq_cpu_hour = st.number_input(
            "CPU Hour", value=0.0, format="%.4f", key="orq_cpu"
        )
        orq_gb_hour = st.number_input(
            "GB Hour", value=0.0, format="%.5f", key="orq_gb"
        )

    orq_pricing = OrqPricing(
        base_subscription=orq_base,
        included_traces=int(orq_included_traces),
        included_storage_gb=orq_included_storage,
        traces_overage_per_k=orq_trace_overage,
        storage_per_gb=orq_storage_price,
        cpu_hour=orq_cpu_hour,
        gb_hour=orq_gb_hour,
    )

    # Currency Conversion
    with st.expander("Currency", expanded=False):
        eur_usd_rate = st.number_input(
            "EUR/USD Rate", value=DEFAULT_EUR_USD_RATE, format="%.2f", key="eur_usd"
        )

# -----------------------------------------------------------------------------
# Main Content: Scenario Configuration
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Scenario Selection")
    preset_name = st.selectbox(
        "Preset",
        options=list(PRESETS.keys()),
        index=0,
    )
    preset = PRESETS[preset_name]

with col2:
    st.subheader("Runtime Parameters")
    runtime_col1, runtime_col2, runtime_col3 = st.columns(3)
    with runtime_col1:
        session_duration = st.number_input(
            "Session Duration (s)", value=10.0, min_value=0.1, format="%.1f"
        )
    with runtime_col2:
        cpu_cores = st.number_input(
            "CPU Cores", value=0.25, min_value=0.01, format="%.2f"
        )
    with runtime_col3:
        memory_gb = st.number_input(
            "Memory (GB)", value=0.5, min_value=0.1, format="%.1f"
        )

runtime = RuntimeParams(
    session_duration_seconds=session_duration,
    cpu_cores=cpu_cores,
    memory_gb=memory_gb,
)

# Scenario Parameters (editable based on preset)
st.subheader("Scenario Parameters")
param_cols = st.columns(4)

with param_cols[0]:
    spans_per_trace = st.number_input(
        "Spans per Trace", value=preset.spans_per_trace, min_value=1
    )
    search_calls = st.number_input(
        "Search Calls per Trace", value=preset.search_calls_per_trace, min_value=0
    )

with param_cols[1]:
    tool_calls = st.number_input(
        "Tool Calls per Trace", value=preset.tool_calls_per_trace, min_value=0
    )
    tokens_per_trace = st.number_input(
        "Tokens per Trace", value=preset.tokens_per_trace, min_value=1000, step=1000
    )

with param_cols[2]:
    bytes_per_token = st.number_input(
        "Bytes per Token", value=preset.bytes_per_token, min_value=1
    )
    total_evals = st.number_input(
        "Total Evaluations", value=preset.total_evaluations, min_value=0, step=1000
    )

with param_cols[3]:
    builtin_evals = st.number_input(
        "Built-in Evals Count", value=preset.builtin_evals_count, min_value=0
    )
    custom_evals = st.number_input(
        "Custom Evals Count", value=preset.custom_evals_count, min_value=0
    )

eval_output_tokens = st.number_input(
    "Eval Output Tokens", value=preset.eval_output_tokens, min_value=1
)

# Memory and Gateway Parameters
st.subheader("Memory & Gateway Parameters")
mem_cols = st.columns(4)

with mem_cols[0]:
    short_term_events = st.number_input(
        "Short-term Events/Trace",
        value=preset.short_term_memory_events_per_trace,
        min_value=0,
    )
with mem_cols[1]:
    long_term_memories = st.number_input(
        "Long-term Memories/Trace",
        value=preset.long_term_memories_per_trace,
        min_value=0.0,
        format="%.2f",
    )
with mem_cols[2]:
    memory_retrievals = st.number_input(
        "Memory Retrievals/Trace",
        value=preset.memory_retrievals_per_trace,
        min_value=0,
    )
with mem_cols[3]:
    tools_indexed = st.number_input(
        "Tools Indexed",
        value=preset.tools_indexed,
        min_value=0,
    )

scenario = ScenarioParams(
    name=preset_name,
    spans_per_trace=spans_per_trace,
    search_calls_per_trace=search_calls,
    tool_calls_per_trace=tool_calls,
    tokens_per_trace=tokens_per_trace,
    bytes_per_token=bytes_per_token,
    total_evaluations=total_evals,
    builtin_evals_count=builtin_evals,
    custom_evals_count=custom_evals,
    eval_output_tokens=eval_output_tokens,
    short_term_memory_events_per_trace=short_term_events,
    long_term_memories_per_trace=long_term_memories,
    memory_retrievals_per_trace=memory_retrievals,
    tools_indexed=tools_indexed,
)

# -----------------------------------------------------------------------------
# Calculate Costs
# -----------------------------------------------------------------------------
aws_costs, orq_costs = calculate_costs_for_trace_range(
    TRACE_POINTS, scenario, runtime, aws_pricing, orq_pricing, eur_usd_rate
)

# -----------------------------------------------------------------------------
# Helper functions for tooltip calculations
# -----------------------------------------------------------------------------
def get_aws_tooltip(traces: int, cost: "CostBreakdown") -> str:
    """Generate detailed tooltip for AWS costs showing formulas and calculations."""
    cpu_hours = (runtime.session_duration_seconds / 3600) * runtime.cpu_cores
    gb_hours = (runtime.session_duration_seconds / 3600) * runtime.memory_gb
    gb_per_trace = scenario.gb_per_trace

    lines = [
        f"<b>AWS Total: €{cost.total:,.2f}</b>",
        "",
        f"<b>Runtime:</b> €{cost.runtime:,.2f}",
        f"  traces × (cpu_hrs × cpu_price + gb_hrs × gb_price) / eur_usd",
        f"  {traces:,} × ({cpu_hours:.6f} × ${aws_pricing.cpu_hour} + {gb_hours:.6f} × ${aws_pricing.gb_hour}) / {eur_usd_rate}",
        "",
        f"<b>Gateway:</b> €{cost.gateway:,.2f}",
        f"  (search_calls × traces / 1M × search_price) + (tool_calls × traces / 1M × tool_price) + tools_indexed_cost",
        f"  ({scenario.search_calls_per_trace} × {traces:,} / 1M × ${aws_pricing.search_tool_per_m}) + ({scenario.tool_calls_per_trace} × {traces:,} / 1M × ${aws_pricing.tool_call_per_m}) + ({scenario.tools_indexed}/100 × ${aws_pricing.tools_indexed_per_100})",
        "",
        f"<b>Memory:</b> €{cost.memory:,.2f}",
        f"  (st_events × traces / 1k × st_price) + (lt_mem × traces / 1k × lt_price) + (retrievals × traces / 1k × ret_price)",
        f"  ({scenario.short_term_memory_events_per_trace} × {traces:,} / 1k × ${aws_pricing.memory_short_term_per_k}) + ({scenario.long_term_memories_per_trace} × {traces:,} / 1k × ${aws_pricing.memory_long_term_per_k}) + ({scenario.memory_retrievals_per_trace} × {traces:,} / 1k × ${aws_pricing.memory_retrieval_per_k})",
        "",
        f"<b>Observability:</b> €{cost.observability:,.2f}",
        f"  (traces × gb_per_trace × span_price) + (0.6 × traces × gb_per_trace × event_price)",
        f"  ({traces:,} × {gb_per_trace:.9f} × ${aws_pricing.span_ingestion_per_gb}) + (0.6 × {traces:,} × {gb_per_trace:.9f} × ${aws_pricing.event_logging_per_gb})",
        "",
        f"<b>Evaluations:</b> €{cost.evaluations:,.2f}",
        f"  builtin_input + builtin_output + custom_evals",
    ]
    return "<br>".join(lines)


def get_orq_tooltip(traces: int, cost: "CostBreakdown") -> str:
    """Generate detailed tooltip for Orq costs showing formulas and calculations."""
    gb_per_trace = scenario.gb_per_trace
    eval_traces = scenario.total_evaluations * (scenario.builtin_evals_count + scenario.custom_evals_count)
    total_storage = traces * gb_per_trace + eval_traces * gb_per_trace
    traces_overage = max(traces - orq_pricing.included_traces, 0)
    storage_overage = max(total_storage - orq_pricing.included_storage_gb, 0)

    lines = [
        f"<b>Orq Total: €{cost.total:,.2f}</b>",
        "",
        f"<b>Base:</b> €{cost.base:,.2f}",
        f"  Base subscription",
        f"  €{orq_pricing.base_subscription}",
        "",
        f"<b>Runtime:</b> €{cost.runtime:,.2f}",
        f"  (Currently €0 - placeholder for future pricing)",
        "",
        f"<b>Gateway:</b> €{cost.gateway:,.2f}",
        f"  (Included in base subscription)",
        "",
        f"<b>Memory:</b> €{cost.memory:,.2f}",
        f"  (Included in base subscription)",
        "",
        f"<b>Observability:</b> €{cost.observability:,.2f}",
        f"  traces_overage_cost + storage_overage_cost",
        f"  Traces overage: max({traces:,} - {orq_pricing.included_traces:,}, 0) = {traces_overage:,}",
        f"  Traces cost: {traces_overage:,} × €{orq_pricing.traces_overage_per_k}/1k = €{traces_overage * orq_pricing.traces_overage_per_k / 1000:,.2f}",
        f"  Storage used: {total_storage:.2f} GB (traces: {traces * gb_per_trace:.2f} + evals: {eval_traces * gb_per_trace:.2f})",
        f"  Storage overage: max({total_storage:.2f} - {orq_pricing.included_storage_gb}, 0) = {storage_overage:.2f} GB",
        f"  Storage cost: {storage_overage:.2f} × €{orq_pricing.storage_per_gb}/GB = €{storage_overage * orq_pricing.storage_per_gb:,.2f}",
        "",
        f"<b>Evaluations:</b> €{cost.evaluations:,.2f}",
        f"  (Storage included in observability calculation above)",
    ]
    return "<br>".join(lines)


# Generate tooltips for each trace point
aws_tooltips = [get_aws_tooltip(t, c) for t, c in zip(TRACE_POINTS, aws_costs)]
orq_tooltips = [get_orq_tooltip(t, c) for t, c in zip(TRACE_POINTS, orq_costs)]

# -----------------------------------------------------------------------------
# Chart 1: Total Cost vs Traces (Line Chart)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Total Cost vs Number of Traces")

df_line = pd.DataFrame(
    {
        "Traces": TRACE_POINTS,
        "AWS": [c.total for c in aws_costs],
        "Orq": [c.total for c in orq_costs],
    }
)

fig_line = go.Figure()
fig_line.add_trace(
    go.Scatter(
        x=df_line["Traces"],
        y=df_line["AWS"],
        mode="lines+markers",
        name="AWS",
        line=dict(color="#FF9900", width=3),
        marker=dict(size=8),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=aws_tooltips,
    )
)
fig_line.add_trace(
    go.Scatter(
        x=df_line["Traces"],
        y=df_line["Orq"],
        mode="lines+markers",
        name="Orq",
        line=dict(color="#6366F1", width=3),
        marker=dict(size=8),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=orq_tooltips,
    )
)

fig_line.update_layout(
    xaxis_title="Number of Traces",
    yaxis_title="Total Cost (EUR)",
    xaxis_type="log",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    margin=dict(l=50, r=50, t=30, b=50),
    hoverlabel=dict(
        bgcolor="white",
        font_size=11,
        font_family="monospace",
        align="left",
    ),
)

st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------------------------------------------------------
# Chart 2: Cost Breakdown (Stacked Bar Chart)
# -----------------------------------------------------------------------------
st.subheader("Cost Breakdown by Component")

# Slider to select trace count for breakdown
breakdown_trace_idx = st.select_slider(
    "Select trace count for breakdown",
    options=list(range(len(TRACE_POINTS))),
    format_func=lambda x: f"{TRACE_POINTS[x]:,}",
    value=len(TRACE_POINTS) // 2,
)
selected_traces = TRACE_POINTS[breakdown_trace_idx]

aws_breakdown = aws_costs[breakdown_trace_idx]
orq_breakdown = orq_costs[breakdown_trace_idx]

components = ["Base", "Runtime", "Gateway", "Memory", "Observability", "Evaluations"]
aws_values = [
    aws_breakdown.base,
    aws_breakdown.runtime,
    aws_breakdown.gateway,
    aws_breakdown.memory,
    aws_breakdown.observability,
    aws_breakdown.evaluations,
]
orq_values = [
    orq_breakdown.base,
    orq_breakdown.runtime,
    orq_breakdown.gateway,
    orq_breakdown.memory,
    orq_breakdown.observability,
    orq_breakdown.evaluations,
]

# Generate component-specific tooltips for the bar chart
def get_component_tooltip(component: str, provider: str, value: float, traces: int) -> str:
    """Generate tooltip for a specific cost component."""
    cpu_hours = (runtime.session_duration_seconds / 3600) * runtime.cpu_cores
    gb_hours = (runtime.session_duration_seconds / 3600) * runtime.memory_gb
    gb_per_trace = scenario.gb_per_trace
    eval_traces = scenario.total_evaluations * (scenario.builtin_evals_count + scenario.custom_evals_count)
    total_storage = traces * gb_per_trace + eval_traces * gb_per_trace
    traces_overage = max(traces - orq_pricing.included_traces, 0)
    storage_overage = max(total_storage - orq_pricing.included_storage_gb, 0)

    if provider == "AWS":
        if component == "Base":
            return f"<b>Base: €{value:,.2f}</b><br>(No base subscription for AWS)"
        elif component == "Runtime":
            return (
                f"<b>Runtime: €{value:,.2f}</b><br><br>"
                f"Formula: traces × (cpu_hrs × cpu_price + gb_hrs × gb_price) / eur_usd<br><br>"
                f"Calculation:<br>"
                f"{traces:,} × ({cpu_hours:.6f} × ${aws_pricing.cpu_hour} + {gb_hours:.6f} × ${aws_pricing.gb_hour}) / {eur_usd_rate}"
            )
        elif component == "Gateway":
            return (
                f"<b>Gateway: €{value:,.2f}</b><br><br>"
                f"Formula: (search × traces/1M × price) + (tools × traces/1M × price) + indexed_cost<br><br>"
                f"Calculation:<br>"
                f"({scenario.search_calls_per_trace} × {traces:,}/1M × ${aws_pricing.search_tool_per_m}) +<br>"
                f"({scenario.tool_calls_per_trace} × {traces:,}/1M × ${aws_pricing.tool_call_per_m}) +<br>"
                f"({scenario.tools_indexed}/100 × ${aws_pricing.tools_indexed_per_100})"
            )
        elif component == "Memory":
            return (
                f"<b>Memory: €{value:,.2f}</b><br><br>"
                f"Formula: (st_events × traces/1k × price) + (lt_mem × traces/1k × price) + (ret × traces/1k × price)<br><br>"
                f"Calculation:<br>"
                f"({scenario.short_term_memory_events_per_trace} × {traces:,}/1k × ${aws_pricing.memory_short_term_per_k}) +<br>"
                f"({scenario.long_term_memories_per_trace} × {traces:,}/1k × ${aws_pricing.memory_long_term_per_k}) +<br>"
                f"({scenario.memory_retrievals_per_trace} × {traces:,}/1k × ${aws_pricing.memory_retrieval_per_k})"
            )
        elif component == "Observability":
            return (
                f"<b>Observability: €{value:,.2f}</b><br><br>"
                f"Formula: (traces × gb/trace × span_price) + (0.6 × traces × gb/trace × event_price)<br><br>"
                f"Calculation:<br>"
                f"({traces:,} × {gb_per_trace:.9f} × ${aws_pricing.span_ingestion_per_gb}) +<br>"
                f"(0.6 × {traces:,} × {gb_per_trace:.9f} × ${aws_pricing.event_logging_per_gb})"
            )
        elif component == "Evaluations":
            return (
                f"<b>Evaluations: €{value:,.2f}</b><br><br>"
                f"Formula: builtin_input + builtin_output + custom_evals<br><br>"
                f"Calculation:<br>"
                f"Built-in: {scenario.total_evaluations:,} × {scenario.builtin_evals_count} evals<br>"
                f"Custom: {scenario.total_evaluations:,} × {scenario.custom_evals_count} × ${aws_pricing.custom_evals_per_k_calls}/1k"
            )
    else:  # Orq
        if component == "Base":
            return f"<b>Base: €{value:,.2f}</b><br><br>Base subscription: €{orq_pricing.base_subscription}"
        elif component == "Runtime":
            return f"<b>Runtime: €{value:,.2f}</b><br><br>(Placeholder - currently €0)"
        elif component == "Gateway":
            return f"<b>Gateway: €{value:,.2f}</b><br><br>(Included in base subscription)"
        elif component == "Memory":
            return f"<b>Memory: €{value:,.2f}</b><br><br>(Included in base subscription)"
        elif component == "Observability":
            return (
                f"<b>Observability: €{value:,.2f}</b><br><br>"
                f"Formula: traces_overage_cost + storage_overage_cost<br><br>"
                f"Traces overage: max({traces:,} - {orq_pricing.included_traces:,}, 0) = {traces_overage:,}<br>"
                f"Traces cost: {traces_overage:,} × €{orq_pricing.traces_overage_per_k}/1k = €{traces_overage * orq_pricing.traces_overage_per_k / 1000:,.2f}<br><br>"
                f"Storage used: {total_storage:.2f} GB<br>"
                f"Storage overage: max({total_storage:.2f} - {orq_pricing.included_storage_gb}, 0) = {storage_overage:.2f} GB<br>"
                f"Storage cost: {storage_overage:.2f} × €{orq_pricing.storage_per_gb}/GB = €{storage_overage * orq_pricing.storage_per_gb:,.2f}"
            )
        elif component == "Evaluations":
            return f"<b>Evaluations: €{value:,.2f}</b><br><br>(Storage included in observability)"
    return f"<b>{component}: €{value:,.2f}</b>"


fig_bar = go.Figure()
colors = ["#3B82F6", "#10B981", "#F59E0B", "#14B8A6", "#EF4444", "#8B5CF6"]

for i, component in enumerate(components):
    aws_tooltip = get_component_tooltip(component, "AWS", aws_values[i], selected_traces)
    orq_tooltip = get_component_tooltip(component, "Orq", orq_values[i], selected_traces)
    fig_bar.add_trace(
        go.Bar(
            name=component,
            x=["AWS", "Orq"],
            y=[aws_values[i], orq_values[i]],
            marker_color=colors[i],
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[aws_tooltip, orq_tooltip],
        )
    )

fig_bar.update_layout(
    barmode="stack",
    yaxis_title="Cost (EUR)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    margin=dict(l=50, r=50, t=30, b=50),
    hoverlabel=dict(
        bgcolor="white",
        font_size=11,
        font_family="monospace",
        align="left",
    ),
)

st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# Summary Table
# -----------------------------------------------------------------------------
st.subheader(f"Cost Summary at {selected_traces:,} Traces")

summary_df = pd.DataFrame(
    {
        "Component": components + ["Total"],
        "AWS (EUR)": [f"€{v:,.2f}" for v in aws_values] + [f"€{aws_breakdown.total:,.2f}"],
        "Orq (EUR)": [f"€{v:,.2f}" for v in orq_values] + [f"€{orq_breakdown.total:,.2f}"],
        "Difference": [f"€{aws_values[i] - orq_values[i]:,.2f}" for i in range(len(components))]
        + [f"€{aws_breakdown.total - orq_breakdown.total:,.2f}"],
    }
)

# Highlight totals
st.dataframe(
    summary_df,
    hide_index=True,
    use_container_width=True,
)

# Savings summary
savings = aws_breakdown.total - orq_breakdown.total
savings_pct = (savings / aws_breakdown.total * 100) if aws_breakdown.total > 0 else 0

col_metric1, col_metric2, col_metric3 = st.columns(3)
with col_metric1:
    st.metric("AWS Total", f"€{aws_breakdown.total:,.2f}")
with col_metric2:
    st.metric("Orq Total", f"€{orq_breakdown.total:,.2f}")
with col_metric3:
    # Use "inverse" when savings is positive (green up arrow), "normal" when negative (red down arrow)
    # But we want: positive savings = green, negative savings = red
    # delta_color "normal" means positive=green, negative=red
    # delta_color "inverse" means positive=red, negative=green
    # Since savings > 0 means Orq is cheaper (good), we want green
    # The delta shows the percentage, which will be positive when AWS > Orq
    delta_color = "normal"  # positive delta = green (savings), negative delta = red (loss)
    st.metric(
        "Savings with Orq",
        f"€{savings:,.2f}",
        delta=f"{savings_pct:.1f}%",
        delta_color=delta_color,
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Note: All costs are estimates based on configured pricing. "
    "AWS costs are converted from USD to EUR using the specified exchange rate."
)
