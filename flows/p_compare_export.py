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

    st.divider()

    tab_within, tab_rerun = st.tabs([
        "Within-run rule toggle analysis",
        "Full re-run with new blocking rules",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Within-run: toggle match_keys off, re-cluster the existing edges
    # This mirrors the linkage-metrics notebook approach: filter df_predict by
    # removing rows belonging to a disabled match_key, then re-cluster.
    # No new model training or prediction is needed — instant results.
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_within:
        st.write(
            "Toggle individual blocking rules on or off below. "
            "The app removes candidate pairs that were **only** captured by disabled "
            "rules, then re-clusters the remaining edges instantly — no re-training needed. "
            "This shows the marginal contribution of each rule."
        )

        # ── Coverage matrix (built once, cached in session) ───────────────────
        cov = st.session_state.get("coverage_matrix")
        if cov is None or cov.empty:
            st.info(
                "Coverage matrix not available. "
                "Run the analysis first (Step 5) to enable within-run comparison."
            )
        else:
            # ── Rule toggles ──────────────────────────────────────────────────
            st.subheader("Toggle blocking rules")
            run1_toggles = run1["run_config"]["blocking_toggles"]

            if "within_toggles" not in st.session_state:
                st.session_state["within_toggles"] = dict(run1_toggles)

            wt = {}
            wt_cols = st.columns(min(4, len(run1_toggles)))
            for i, (field, was_on) in enumerate(run1_toggles.items()):
                col = wt_cols[i % len(wt_cols)]
                wt[field] = col.toggle(
                    field, value=st.session_state["within_toggles"].get(field, was_on),
                    key=f"wt_{field}",
                )
            st.session_state["within_toggles"] = wt

            # ── Filter + re-cluster ───────────────────────────────────────────
            from modules.splink_runner import filter_predict_by_active_rules, recluster_filtered

            filtered_df = filter_predict_by_active_rules(
                run1["df_predict"], cov, wt
            )
            n_filt = len(filtered_df)
            n_orig = len(run1["df_predict"])

            removed = n_orig - n_filt
            st.caption(
                f"Edges after toggle: **{n_filt:,}** "
                f"(removed {removed:,} = {100*removed/max(n_orig,1):.1f}% of Run 1 edges)"
            )

            if st.button("Re-cluster with toggled rules", type="primary", key="within_recluster"):
                with st.spinner("Re-clustering…"):
                    threshold = run1["run_config"].get("cluster_threshold", 0.8)
                    new_clusters = recluster_filtered(
                        filtered_df,
                        st.session_state["fakea"],
                        st.session_state.get("fakeb"),
                        threshold=threshold,
                    )
                    st.session_state["within_clusters"] = new_clusters

            within_clusters = st.session_state.get("within_clusters")
            if within_clusters is not None and not within_clusters.empty:
                n_new_cl = within_clusters["cluster_id"].nunique()
                n_orig_cl = m1["n_clusters"]

                wk1, wk2, wk3 = st.columns(3)
                wk1.metric("Edges (toggled)", f"{n_filt:,}",
                           delta=f"{n_filt - n_orig:+,}")
                wk2.metric("Clusters (re-clustered)", f"{n_new_cl:,}",
                           delta=f"{n_new_cl - n_orig_cl:+,}")
                wk3.metric("Edges removed", f"{removed:,}")

                # Set-difference metrics (from the linkage-metrics notebook pattern)
                import duckdb as _ddb
                _con = _ddb.connect()
                _con.register("orig_edges", run1["df_predict"][
                    ["unique_id_l","unique_id_r","source_dataset_l","source_dataset_r"]
                ])
                _con.register("filt_edges", filtered_df[
                    ["unique_id_l","unique_id_r","source_dataset_l","source_dataset_r"]
                ] if not filtered_df.empty else run1["df_predict"].iloc[0:0][
                    ["unique_id_l","unique_id_r","source_dataset_l","source_dataset_r"]
                ])

                shared_n = _con.sql("""
                    SELECT COUNT(*) FROM orig_edges o
                    INNER JOIN filt_edges f
                    USING (unique_id_l, unique_id_r, source_dataset_l, source_dataset_r)
                """).fetchone()[0]
                removed_n = n_orig - shared_n
                _con.close()

                st.write("**Set-difference edge metrics**")
                st.dataframe(pd.DataFrame([
                    {"Metric": "Edges in original run", "Count": n_orig},
                    {"Metric": "Edges retained after toggle", "Count": shared_n},
                    {"Metric": "Edges removed by disabling rules", "Count": removed_n},
                    {"Metric": "Clusters (original)", "Count": n_orig_cl},
                    {"Metric": "Clusters (after toggle)", "Count": n_new_cl},
                    {"Metric": "Cluster delta", "Count": n_new_cl - n_orig_cl},
                ]), use_container_width=True, hide_index=True)

                st.caption(
                    "Interpretation: 'Edges removed' shows how many candidate pairs "
                    "were contributed exclusively by the disabled rule(s). "
                    "A large number means that rule was capturing many unique pairs."
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Full re-run (original behaviour, kept for complete model comparison)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_rerun:
        st.write(
            "Run a completely new linkage model with different blocking rules. "
            "This re-trains (for probabilistic) or re-predicts (deterministic) "
            "from scratch, giving a fully independent second result to compare."
        )
        st.subheader("Run 1 summary")
        active1 = [f for f, v in run1["run_config"]["blocking_toggles"].items() if v]
        st.caption(f"Blocking rules: {', '.join(active1)}")
        _metric_cards([
            ("Run 1: Edges",    f"{m1['n_edges']:,}"),
            ("Run 1: Clusters", f"{m1['n_clusters']:,}"),
            ("Run 1: Mean match prob",
             str(m1["match_prob_stats"]["mean_match_prob"].iloc[0])
             if not m1["match_prob_stats"].empty else "N/A"),
        ])

        st.divider()
        st.subheader("Modify blocking rules for Run 2")

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

        # ── Composite blocking rules for Run 2 ────────────────────────────────
        with st.expander("Composite blocking rules for Run 2 (optional)", expanded=False):
            st.write("Combine two or three fields into a single AND rule.")
            if "r2_composite_rules" not in st.session_state:
                st.session_state["r2_composite_rules"] = {}
            sel_fields = st.session_state.get("selected_fields", [])
            if len(sel_fields) >= 2:
                rc1, rc2, rc3, rc4 = st.columns([2, 2, 2, 1])
                rf1 = rc1.selectbox("Field 1", sel_fields, key="r2_cb_f1")
                rf2_opts = [f for f in sel_fields if f != rf1]
                rf2 = rc2.selectbox("Field 2", rf2_opts, key="r2_cb_f2")
                rf3_opts = ["(none)"] + [f for f in sel_fields if f not in (rf1, rf2)]
                rf3_sel = rc3.selectbox("Field 3 (optional)", rf3_opts, key="r2_cb_f3")
                rf3 = None if rf3_sel == "(none)" else rf3_sel
                if rc4.button("Add rule", key="r2_cb_add"):
                    rkey = f"{rf1}+{rf2}" + (f"+{rf3}" if rf3 else "")
                    st.session_state["r2_composite_rules"][rkey] = True
            for rkey in list(st.session_state.get("r2_composite_rules", {}).keys()):
                rparts = rkey.split("+")
                rsql = " AND ".join(f'l."{p}" = r."{p}"' for p in rparts)
                rcr1, rcr2 = st.columns([4, 1])
                rcr1.code(rsql)
                if rcr2.button("Remove", key=f"r2_rm_{rkey}"):
                    del st.session_state["r2_composite_rules"][rkey]

        has_single_r2 = any(r2_toggles.values())
        has_composite_r2 = bool(st.session_state.get("r2_composite_rules"))
        if not has_single_r2 and not has_composite_r2:
            st.error("At least one blocking rule (single-field or composite) must be defined for Run 2.")
        else:
            st.session_state["run2_blocking_toggles"] = r2_toggles

            if st.button("Run full analysis with updated blocking rules", type="primary"):
                fakea_r2 = st.session_state.get("fakea")
                if fakea_r2 is not None:
                    n_a = len(fakea_r2)
                    fakeb_r2 = st.session_state.get("fakeb")
                    n_b = len(fakeb_r2) if fakeb_r2 is not None else n_a
                    for _key, _en in r2_toggles.items():
                        if not _en or "+" in _key or _key not in fakea_r2.columns:
                            continue
                        _n_u = fakea_r2[_key].nunique()
                        if _n_u == 0:
                            continue
                        _est = int((n_a / _n_u) * (n_b / _n_u) * _n_u)
                        if _est > 5_000_000:
                            st.error(
                                f"Blocking on `{_key}` would generate ~{_est:,} pairs "
                                f"({_n_u} unique values). Disable this rule."
                            )
                            st.stop()

                with st.spinner("Running Run 2…"):
                    try:
                        run2 = run_linkage(
                            fakea=st.session_state["fakea"],
                            fakeb=st.session_state["fakeb"],
                            selected_fields=st.session_state["selected_fields"],
                            blocking_toggles=r2_toggles,
                            operation_mode=st.session_state["operation_mode"],
                            linkage_type=st.session_state["linkage_type"],
                            hyperparams=st.session_state.get("hyperparams", {}),
                            composite_rules=st.session_state.get("r2_composite_rules", {}),
                        )
                        m2 = compute_intra_metrics(run2["df_predict"], run2["df_cluster"])
                        st.session_state["run2_results"] = run2
                        st.session_state["run2_metrics"] = m2
                        st.success("Run 2 complete.")
                    except Exception as e:
                        st.error(f"Run 2 failed: {e}")

        if st.session_state.get("run2_results") is not None:
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
            kc1.metric("Edges", f"{m2['n_edges']:,}",
                       delta=f"{m2['n_edges'] - m1['n_edges']:+,}")
            kc2.metric("Clusters", f"{m2['n_clusters']:,}",
                       delta=f"{m2['n_clusters'] - m1['n_clusters']:+,}")
            kc3.metric("Mean match prob", f"{mp2:.4f}",
                       delta=f"{mp2 - mp1:+.4f}")

            st.divider()
            ed = inter.get("edge_diff", pd.DataFrame())
            if not ed.empty:
                st.write("**Edge Changes Between Runs**")
                ed_d = ed.set_index("category")["n"].to_dict()
                st.dataframe(pd.DataFrame([
                    {"Metric": "Shared edges (both runs)",    "Count": ed_d.get("shared", 0)},
                    {"Metric": "Edges added in Run 2",        "Count": ed_d.get("added", 0)},
                    {"Metric": "Edges removed in Run 2",      "Count": ed_d.get("removed", 0)},
                    {"Metric": "Exact matching clusters",     "Count": inter.get("n_exact_matching_clusters", 0)},
                    {"Metric": "Partially matching clusters", "Count": inter.get("n_partial_matching_clusters", 0)},
                ]), use_container_width=True, hide_index=True)

            pd1 = inter.get("prob_dist_run1", pd.DataFrame())
            pd2 = inter.get("prob_dist_run2", pd.DataFrame())
            if not pd1.empty and not pd2.empty:
                pd1["run"] = "Run 1"; pd2["run"] = "Run 2"
                fig = px.bar(pd.concat([pd1, pd2]), x="prob_bin", y="n_edges", color="run",
                             barmode="group", title="Match Probability Distribution",
                             template="simple_white",
                             color_discrete_sequence=["#1E6EC4", "#E55C30"])
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True)

            cs1 = inter.get("cluster_sizes_run1", pd.DataFrame())
            cs2 = inter.get("cluster_sizes_run2", pd.DataFrame())
            if not cs1.empty and not cs2.empty:
                cs1["run"] = "Run 1"; cs2["run"] = "Run 2"
                fig2 = px.bar(pd.concat([cs1, cs2]), x="n_nodes", y="n_clusters", color="run",
                              barmode="group", title="Cluster Size Distribution",
                              template="simple_white",
                              color_discrete_sequence=["#1E6EC4", "#E55C30"])
                fig2.update_layout(height=340)
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()
            if st.button("Generate PDF report for Run 2"):
                with st.spinner("Generating…"):
                    try:
                        pdf2 = generate_report(
                            run_label="Run 2", run_config=run2["run_config"], metrics=m2,
                            n_input_records=run2["n_input_records"],
                            model_params=run2.get("model_params", {}),
                            missingness_a=run2.get("missingness_a", {}),
                            missingness_b=run2.get("missingness_b", {}),
                            blocking_counts=run2.get("blocking_counts", []),
                            unlinkables=run2.get("unlinkables", {}),
                            settings_used=run2.get("settings_used", {}),
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
        "Cohort CSV downloaded. Future versions will support direct databank integration."
    )