# utils/helpers.py
# Reusable UI components and the core analysis runner.
# _run_analysis_and_store() is the single point through which ALL three flows
# (standard, upload, advanced) trigger a Splink linkage run.

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.splink_runner import run_linkage, build_coverage_matrix
from modules.metrics_engine import (
    compute_intra_metrics, compute_confusion_matrix,
    compute_truth_space, compute_crl_score,
)


def _metric_cards(metrics: list) -> None:
    """Render a row of st.metric cards from [(label, value), ...] tuples."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label=label, value=value)


def _plotly_bar(df: pd.DataFrame, x: str, y: str, title: str,
                colour: str = "#1E6EC4") -> "go.Figure":
    """Return a clean Plotly bar chart."""
    fig = px.bar(df, x=x, y=y, title=title,
                 color_discrete_sequence=[colour], template="simple_white")
    fig.update_layout(
        title_font_size=14,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        margin=dict(l=40, r=20, t=50, b=40), height=320,
    )
    return fig


def _run_analysis_and_store(
    fakea, fakeb, selected_fields, blocking_toggles,
    operation_mode, linkage_type, hyperparams, composite_rules,
) -> bool:
    """Run Splink linkage and persist ALL results to session state.

    Validates selected_fields and blocking_toggles against actual dataset
    columns before calling run_linkage.  Auto-repairs mismatches with a
    visible warning so the user always gets a result rather than a crash.

    Returns True on success, False if validation or linkage failed.
    """
    # ── Field validation: find columns common to all input datasets ────────────
    cols_a = set(fakea.columns)
    cols_b = set(fakeb.columns) if fakeb is not None else cols_a
    common  = cols_a & cols_b

    valid_fields   = [f for f in selected_fields if f in common]
    skipped_fields = [f for f in selected_fields if f not in common]

    if skipped_fields:
        st.warning(
            f"Fields not found in all datasets — skipped: **{', '.join(skipped_fields)}**"
        )
    if not valid_fields:
        st.error(
            "None of the selected fields exist in the loaded dataset. "
            "Go back to Configure Fields and select columns that are present in your data. "
            f"Available columns (Dataset A): {sorted(cols_a)}"
        )
        return False

    # ── Blocking validation: keep only rules whose column is in common set ─────
    valid_blocking = {}
    for key, enabled in blocking_toggles.items():
        if not enabled:
            valid_blocking[key] = False
        elif "+" in key:
            parts = [f.strip() for f in key.split("+")]
            valid_blocking[key] = all(p in common for p in parts)
        else:
            valid_blocking[key] = key in common

    if not any(valid_blocking.values()):
        st.error(
            "No active blocking rules match columns in the dataset. "
            f"Columns present in both datasets: {sorted(common)}"
        )
        return False

    # ── Comparison types (upload flow only; None = use default comparisons) ────
    raw_ct   = st.session_state.get("upload_comp_types") or {}
    comp_types = {f: t for f, t in raw_ct.items() if f in valid_fields} or None

    # ── Run linkage ────────────────────────────────────────────────────────────
    try:
        results = run_linkage(
            fakea=fakea, fakeb=fakeb,
            selected_fields=valid_fields,
            blocking_toggles=valid_blocking,
            operation_mode=operation_mode,
            linkage_type=linkage_type,
            hyperparams=hyperparams,
            composite_rules=composite_rules,
            comp_types=comp_types,
        )
    except Exception as e:
        st.error(f"Linkage failed: {e}")
        return False

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_intra_metrics(results["df_predict"], results["df_cluster"])
    cm      = compute_confusion_matrix(results["df_predict"], fakea, fakeb, operation_mode)

    if linkage_type == "probabilistic":
        ts  = compute_truth_space(results["df_predict"], fakea, fakeb, operation_mode)
        crl = compute_crl_score(ts)
    else:
        ts  = None
        crl = {}

    cov = build_coverage_matrix(results["df_predict"], valid_fields)

    # ── Persist ────────────────────────────────────────────────────────────────
    st.session_state.update({
        "run1_results":    results,
        "run1_metrics":    metrics,
        "run1_cm":         cm,
        "run1_ts":         ts,
        "run1_crl":        crl,
        "coverage_matrix": cov,
        "explorer_toggles": dict(valid_blocking),
    })
    return True
