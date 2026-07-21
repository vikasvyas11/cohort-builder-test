# pages/p_compare_export.py
# Comparison page (Run 1 vs Run 2) and Export page.
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# Local core engine imports
from modules.metrics_engine import compute_inter_metrics, compute_intra_metrics
from modules.report_gen import generate_report
from modules.splink_runner import run_linkage  # FIXED: Imported run_linkage explicitly to resolve NameError
from utils.helpers import _metric_cards, _plotly_bar, _run_analysis_and_store
from utils.nav import _back_button, _go_to


def page_comparison():
    _back_button()
    st.title("Step 6: Compare Runs")

    if st.session_state.get("run1_results") is None:
        st.warning("No Run 1 results. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    run1 = st.session_state["run1_results"]
    m1   = st.session_state["run1_metrics"]

    st.write(
        "Modify blocking rules and re-run to compare how the results change. "
        "Both runs use the same operation mode and linkage type."
    )
    st.divider()

    # Run 1 summary
    st.subheader("Run 1 summary")
    active1 = [f for f, v in run1["run_config"]["blocking_toggles"].items() if v]
    st.caption(f"Blocking rules: {', '.join(active1)}")
    _metric_cards([
        ("Run 1: Edges",         f"{m1['n_edges']:,}"),
        ("Run 1: Clusters",      f"{m1['n_clusters']:,}"),
        ("Run 1: Mean match prob",
         str(m1["match_prob_stats"]["mean_match_prob"].iloc[0])
         if not m1["match_prob_stats"].empty else "N/A"),
    ])

    st.divider()
    st.subheader("Modify blocking rules for Run 2")

    # Initialise Run 2 toggles from Run 1 if not yet set
    if st.session_state.get("run2_blocking_toggles") is None:
        st.session_state["run2_blocking_toggles"] = dict(
            run1["run_config"]["blocking_toggles"]
        )

    r2_toggles = {}
    tc = st.columns(3)
    for i, field in enumerate(st.session_state["selected_fields"]):
        col = tc[i % 3]
        enabled = col.toggle(
            field,
            value=st.session_state["run2_blocking_toggles"].get(field, True),
            key=f"r2_{field}",
        )
        r2_toggles[field] = enabled

    if not any(r2_toggles.values()):
        st.error("At least one blocking rule must be active for Run 2.")
        return
    st.session_state["run2_blocking_toggles"] = r2_toggles

    if st.button("Run analysis with updated blocking rules", type="primary"):
        with st.spinner("Running Run 2..."):
            try:
                # Executes the model iteration with your user configurations
                run2 = run_linkage(
                    fakea=st.session_state["fakea"],
                    fakeb=st.session_state["fakeb"],
                    selected_fields=st.session_state["selected_fields"],
                    blocking_toggles=r2_toggles,
                    operation_mode=st.session_state["operation_mode"],
                    linkage_type=st.session_state["linkage_type"],
                    hyperparams=st.session_state.get("hyperparams", {}),
                )
                m2 = compute_intra_metrics(run2["df_predict"], run2["df_cluster"])
                st.session_state["run2_results"] = run2
                st.session_state["run2_metrics"] = m2
                st.success("Run 2 complete.")
            except Exception as e:
                st.error(f"Run 2 failed: {e}")
                return

    if st.session_state.get("run2_results") is None:
        return

    run2 = st.session_state["run2_results"]
    m2   = st.session_state["run2_metrics"]

    inter = compute_inter_metrics(
        run1["df_predict"], run2["df_predict"],
        run1["df_cluster"], run2["df_cluster"],
    )

    st.divider()
    st.subheader("Comparison: Run 1 vs Run 2")

    mp1 = (m1["match_prob_stats"]["mean_match_prob"].iloc[0]
           if not m1["match_prob_stats"].empty else 0)
    mp2 = (m2["match_prob_stats"]["mean_match_prob"].iloc[0]
           if not m2["match_prob_stats"].empty else 0)

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Edges",            f"{m2['n_edges']:,}",
               delta=f"{m2['n_edges'] - m1['n_edges']:+,}")
    kc2.metric("Clusters",         f"{m2['n_clusters']:,}",
               delta=f"{m2['n_clusters'] - m1['n_clusters']:+,}")
    kc3.metric("Mean match prob",  f"{mp2:.4f}",
               delta=f"{mp2 - mp1:+.4f}")

    st.divider()

    # Edge difference table
    ed = inter.get("edge_diff", pd.DataFrame())
    if not ed.empty:
        st.write("**Edge Changes Between Runs**")
        ed_d = ed.set_index("category")["n"].to_dict()
        st.dataframe(pd.DataFrame([
            {"Metric":"Shared edges (both runs)",    "Count": ed_d.get("shared",0)},
            {"Metric":"Edges added in Run 2",        "Count": ed_d.get("added",0)},
            {"Metric":"Edges removed in Run 2",      "Count": ed_d.get("removed",0)},
            {"Metric":"Exact matching clusters",     "Count": inter.get("n_exact_matching_clusters",0)},
            {"Metric":"Partially matching clusters", "Count": inter.get("n_partial_matching_clusters",0)},
        ]), use_container_width=True, hide_index=True)

    # Side-by-side probability distribution
    pd1 = inter.get("prob_dist_run1", pd.DataFrame())
    pd2 = inter.get("prob_dist_run2", pd.DataFrame())
    if not pd1.empty and not pd2.empty:
        pd1["run"] = "Run 1"
        pd2["run"] = "Run 2"
        fig = px.bar(
            pd.concat([pd1, pd2]), x="prob_bin", y="n_edges", color="run",
            barmode="group", title="Match Probability Distribution Comparison",
            template="simple_white",
            color_discrete_sequence=["#1E6EC4","#E55C30"],
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster size comparison
    cs1 = inter.get("cluster_sizes_run1", pd.DataFrame())
    cs2 = inter.get("cluster_sizes_run2", pd.DataFrame())
    if not cs1.empty and not cs2.empty:
        cs1["run"] = "Run 1"
        cs2["run"] = "Run 2"
        fig2 = px.bar(
            pd.concat([cs1, cs2]), x="n_nodes", y="n_clusters", color="run",
            barmode="group", title="Cluster Size Distribution Comparison",
            template="simple_white",
            color_discrete_sequence=["#1E6EC4","#E55C30"],
        )
        fig2.update_layout(height=340)
        st.plotly_chart(fig2, use_container_width=True)

    # PDF for Run 2
    st.divider()
    if st.button("Generate PDF report for Run 2"):
        with st.spinner("Generating..."):
            try:
                pdf2 = generate_report(
                    run_label="Run 2",
                    run_config=run2["run_config"], metrics=m2,
                    n_input_records=run2["n_input_records"],
                    model_params=run2.get("model_params",{}),
                    missingness_a=run2.get("missingness_a",{}),
                    missingness_b=run2.get("missingness_b",{}),
                    blocking_counts=run2.get("blocking_counts",[]),
                    unlinkables=run2.get("unlinkables",{}),
                    settings_used=run2.get("settings_used",{}),
                )
                st.download_button("Download Run 2 PDF", data=pdf2,
                                   file_name=f"linkage_report_run2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   mime="application/pdf")
            except Exception as e:
                st.error(f"PDF failed: {e}")

    st.divider()
    if st.button("Continue to export", type="primary"):
        _go_to(6)


def page_export():
    _back_button()
    st.title("Step 7: Export Cohort")

    if st.session_state.get("run1_results") is None:
        st.warning("No analysis results available. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    st.write(
        "Download the final cohort as a CSV. The output contains all original "
        "record fields plus a cluster_id column. Records sharing the same "
        "cluster_id are predicted to represent the same real-world individual."
    )
    st.divider()

    st.subheader("Select which run to export")
    run_opts = ["Run 1"]
    if st.session_state.get("run2_results") is not None:
        run_opts.append("Run 2")
    else:
        st.caption("Run 2 is not available. Complete a comparison run to add it as an option.")

    selected_run = st.radio(
        "Export cluster assignments from:", run_opts, horizontal=True
    )
    chosen = (
        st.session_state["run1_results"]
        if selected_run == "Run 1"
        else st.session_state["run2_results"]
    )

    st.divider()

    df_cluster = chosen["df_cluster"]
    op_mode    = chosen["run_config"]["operation_mode"]

    if op_mode == "dedupe":
        raw = st.session_state["fakea"].copy()
    else:
        raw = pd.concat(
            [st.session_state["fakea"], st.session_state["fakeb"]],
            ignore_index=True,
        )

    merge_keys = (
        ["unique_id", "source_dataset"]
        if "source_dataset" in df_cluster.columns
        else ["unique_id"]
    )
    cohort = raw.merge(
        df_cluster[merge_keys + ["cluster_id"]],
        on=merge_keys, how="left",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records",          f"{len(cohort):,}")
    c2.metric("Distinct cluster IDs",   f"{cohort['cluster_id'].nunique():,}")
    c3.metric("Records with cluster",   f"{cohort['cluster_id'].notna().sum():,}")

    st.subheader("Cohort preview (first 50 rows, sorted by cluster_id)")
    st.dataframe(cohort.sort_values("cluster_id").head(50),
                 use_container_width=True)

    st.divider()
    csv_bytes = cohort.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download cohort CSV ({selected_run})",
        data=csv_bytes,
        file_name=f"cohort_{selected_run.lower().replace(' ','_')}.csv",
        mime="text/csv",
    )

    st.info(
        "In the full SAIL deployment version, this page will include direct "
        "provisioning to the SAIL Databank. For this MVP, CSV download only."
    )