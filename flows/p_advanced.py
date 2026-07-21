# pages/p_advanced.py
# Advanced flow: upload a pre-trained Splink model JSON and jump straight
# to prediction, skipping all EM training.

import io
import json
import urllib.request

import pandas as pd
import streamlit as st
from datetime import datetime

from modules.data_builder import build_datasets
from modules.splink_runner import run_linkage_from_json
from modules.metrics_engine import (
    compute_intra_metrics, compute_confusion_matrix,
    compute_truth_space, compute_crl_score,
)
from modules.splink_runner import build_coverage_matrix
from utils.nav import _back_button, _go_to
from utils.state import clear_run_results


def page_advanced_setup() -> None:
    _back_button("Back to landing")
    st.title("Advanced Setup: Upload Pre-trained Model JSON")
    st.write(
        "Upload a Splink 4.x model JSON (output of linker.misc.save_model_to_json()). "
        "Prediction runs directly from the trained probabilities — no EM training."
    )
    st.divider()

    # ── JSON upload ────────────────────────────────────────────────────────────
    st.subheader("1. Upload Model JSON")
    uploaded = st.file_uploader("Splink model JSON", type=["json"],
                                 help="Produced by linker.misc.save_model_to_json(). "
                                      "Must contain trained m/u probabilities.")
    if uploaded:
        try:
            model_json = json.loads(uploaded.read())
            st.session_state["advanced_json"] = model_json
            comps   = model_json.get("comparisons", [])
            brs     = model_json.get("blocking_rules_to_generate_predictions", [])
            st.success(f"JSON loaded: {len(comps)} comparisons, {len(brs)} blocking rules.")
            with st.expander("Summary", expanded=False):
                st.write(f"**Link type:** {model_json.get('link_type','?')}")
                st.write(f"**Fields:** {', '.join(c.get('output_column_name','?') for c in comps)}")
        except Exception as e:
            st.error(f"Cannot parse JSON: {e}")

    st.divider()

    # ── Dataset selection ──────────────────────────────────────────────────────
    st.subheader("2. Dataset")
    if not st.session_state["dataset_ready"]:
        if st.button("Load dummy dataset (fake1000)", type="primary"):
            with st.spinner("Building datasets..."):
                try:
                    _, fakea, fakeb = build_datasets()
                    st.session_state["fakea"]         = fakea
                    st.session_state["fakeb"]         = fakeb
                    st.session_state["dataset_ready"] = True
                    st.success("Dummy dataset loaded.")
                except Exception as e:
                    st.error(str(e))
    else:
        fakea = st.session_state["fakea"]
        st.success(f"Dataset A loaded: {len(fakea):,} records.")

    st.divider()

    # ── Operation mode + threshold ─────────────────────────────────────────────
    st.subheader("3. Operation mode")
    op = st.radio("Mode:", ["dedupe", "link_dedupe"],
                  format_func=lambda x: "Deduplication only" if x == "dedupe"
                                        else "Link and deduplicate",
                  horizontal=True,
                  index=0 if st.session_state["advanced_op_mode"] == "dedupe" else 1)
    st.session_state["advanced_op_mode"] = op
    threshold = st.slider("Cluster probability threshold", 0.5, 0.99, 0.8, 0.01)

    st.divider()

    model_json = st.session_state.get("advanced_json")
    ready      = model_json and st.session_state["dataset_ready"]
    if not ready:
        st.info("Upload a JSON file and load a dataset to continue.")

    if ready and st.button("Run prediction from uploaded model", type="primary"):
        with st.spinner("Running prediction (no training)..."):
            try:
                fakea = st.session_state["fakea"]
                fakeb = st.session_state["fakeb"] if op == "link_dedupe" else None
                results = run_linkage_from_json(model_json, fakea, fakeb, op, threshold)
                metrics = compute_intra_metrics(results["df_predict"], results["df_cluster"])
                cm      = compute_confusion_matrix(results["df_predict"], fakea, fakeb, op)
                ts      = compute_truth_space(results["df_predict"], fakea, fakeb, op)
                crl     = compute_crl_score(ts)
                fields  = results["run_config"]["selected_fields"]
                cov     = build_coverage_matrix(results["df_predict"], fields)

                clear_run_results()
                st.session_state.update({
                    "run1_results":    results,
                    "run1_metrics":    metrics,
                    "run1_cm":         cm,
                    "run1_ts":         ts,
                    "run1_crl":        crl,
                    "coverage_matrix": cov,
                    "explorer_toggles": dict(results["run_config"]["blocking_toggles"]),
                    "operation_mode":  op,
                    "linkage_type":    "probabilistic",
                })
                st.success(
                    f"Prediction complete: {results['n_edges']:,} edges, "
                    f"{results['n_clusters']:,} clusters."
                )
                _go_to(4)
            except Exception as e:
                st.error(str(e))
