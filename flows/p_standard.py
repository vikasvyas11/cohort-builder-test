# pages/p_standard.py
# Standard flow: Configure, Operation Mode, Linkage Type.
import streamlit as st
from utils.state import ALL_FIELDS
from utils.nav import _back_button, _go_to



def page_configure() -> None:
    _back_button()
    st.title("Step 2: Configure Fields and Blocking Rules")

    if not st.session_state["dataset_ready"]:
        st.warning("Please load a dataset first.")
        if st.button("Go to landing"):
            _go_to(0)
        return

    with st.expander("Dataset A preview", expanded=False):
        st.dataframe(st.session_state["fakea"].head(10), use_container_width=True)
    st.divider()

    st.subheader("Fields to include in comparisons")
    st.write(
        "Select which fields to compare. unique_id and cluster are excluded "
        "as they are identifiers, not linkage features."
    )
    field_cols = st.columns(2)
    selected_fields = []
    for i, field in enumerate(ALL_FIELDS):
        col = field_cols[i % 2]
        if col.checkbox(field,
                        value=(field in st.session_state["selected_fields"]),
                        key=f"field_{field}"):
            selected_fields.append(field)

    if not selected_fields:
        st.error("At least one field must be selected.")
        return
    st.session_state["selected_fields"] = selected_fields
    st.divider()

    st.subheader("Single-field blocking rules")
    st.write(
        "Each toggle creates one independent blocking rule. "
        "Two records are compared only if they agree exactly on at least one "
        "active blocking field."
    )
    blocking_toggles = {}
    t_cols = st.columns(3)
    for i, field in enumerate(selected_fields):
        enabled = t_cols[i % 3].toggle(
            field,
            value=st.session_state["blocking_toggles"].get(field, True),
            key=f"block_{field}",
        )
        blocking_toggles[field] = enabled

    has_single    = any(blocking_toggles.values())
    has_composite = bool(st.session_state.get("composite_rules"))
    if not has_single and not has_composite:
        st.error("At least one blocking rule (single-field or composite) must be defined.")
        return
    st.session_state["blocking_toggles"] = blocking_toggles
    st.caption(f"Active rules: {', '.join(f for f, v in blocking_toggles.items() if v)}")

    with st.expander("Composite blocking rules (advanced)", expanded=False):
        st.write(
            "Combine two or three fields into a single AND rule. "
            "Composite-only configurations (no single-field rules toggled on) are valid."
        )
        cb1, cb2, cb3, cb4 = st.columns([2, 2, 2, 1])
        f1 = cb1.selectbox("Field 1", selected_fields, key="cb_f1")
        f2_opts = [f for f in selected_fields if f != f1]
        f2 = cb2.selectbox("Field 2", f2_opts, key="cb_f2") if f2_opts else None
        f3_opts = ["(none)"] + [f for f in selected_fields if f not in (f1, f2)]
        f3_sel  = cb3.selectbox("Field 3 (optional)", f3_opts, key="cb_f3")
        f3 = None if f3_sel == "(none)" else f3_sel
        if cb4.button("Add", key="cb_add") and f2:
            rule_key = f"{f1}+{f2}" + (f"+{f3}" if f3 else "")
            st.session_state["composite_rules"][rule_key] = True
        for key in list(st.session_state.get("composite_rules", {}).keys()):
            parts = key.split("+")
            sql_parts = " AND ".join(f'l."{p}" = r."{p}"' for p in parts)
            cr1, cr2 = st.columns([4, 1])
            cr1.code(sql_parts)
            if cr2.button("Remove", key=f"rm_{key}"):
                del st.session_state["composite_rules"][key]

    with st.expander("Training hyperparameters (probabilistic mode only)", expanded=False):
        hp = st.session_state.get("hyperparams", {})
        nhp = {}
        nhp["max_iterations"] = st.number_input(
            "Max EM iterations", 5, 500,
            value=hp.get("max_iterations", 25), step=5)
        nhp["em_convergence"] = st.number_input(
            "EM convergence", 1e-8, 0.01,
            value=hp.get("em_convergence", 0.0001), format="%.8f")
        nhp["recall_estimate"] = st.slider(
            "Recall estimate for prior", 0.1, 0.99,
            value=hp.get("recall_estimate", 0.6), step=0.05)
        st.session_state["hyperparams"] = nhp

    st.divider()
    if st.button("Continue to operation mode", type="primary"):
        _go_to(2)


def page_operation() -> None:
    _back_button()
    flow = st.session_state.get("flow", "standard")
    fakea = st.session_state.get("fakea")

    # Track dynamic error configuration rules
    if "custom_error_rates" not in st.session_state:
        st.session_state["custom_error_rates"] = {}

    n_a = f"{len(fakea):,}" if fakea is not None else "N/A"

    st.title("Step 3: Operation Mode")
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Deduplication only")
        st.write(
            "Examine Dataset A and identify internal duplicates. "
            "Use when you have one dataset and want to remove duplicate records."
        )
        st.write(f"**Dataset used:** Dataset A ({n_a} records)")
        if st.button("Select: Deduplication only", use_container_width=True, type="primary"):
            st.session_state["operation_mode"] = "dedupe"
            st.session_state["fakeb"] = None
            _go_to(3)

    with c2:
        st.subheader("Link and deduplicate")
        st.write(
            "Link Dataset A with Dataset B across two separate sources. "
            "Dataset B is a 50% sample of Dataset A with errors introduced during "
            "the EDA step (or via the Upload flow for custom datasets)."
        )

        std_fb = st.session_state.get("std_fakeb")   # pre-built fakeb from data_builder
        current_fb = st.session_state.get("fakeb")

        if std_fb is not None and current_fb is None:
            # Auto-use the staged fakeb from build_datasets()
            st.info(
                f"Dataset B available: {len(std_fb):,} records "
                "(50% sample of Dataset A with controlled errors from data builder)."
            )
            st.write(
                "To customise error rates for Dataset B, go to the "
                "**EDA and Cleaning** page in the Upload flow."
            )
            if st.button("Use pre-built Dataset B", use_container_width=True):
                st.session_state["fakeb"] = std_fb
                st.rerun()

        elif current_fb is not None:
            n_b = f"{len(current_fb):,}"
            st.success(f"Dataset B ready: {n_b} records.")
        else:
            st.warning(
                "Dataset B not yet available. Load the dummy dataset from the "
                "landing page, or use the Upload flow to generate a custom Dataset B."
            )

        current_fb = st.session_state.get("fakeb")
        btn_disabled = current_fb is None
        if st.button("Select: Link and deduplicate", use_container_width=True,
                     type="primary", disabled=btn_disabled):
            st.session_state["operation_mode"] = "link_dedupe"
            _go_to(3)


def page_linkage_type() -> None:
    _back_button()
    st.title("Step 4: Linkage Type")
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Deterministic")
        st.write(
            "Records declared a match if they satisfy at least one active blocking rule. "
            "No training. All matched pairs get match_probability = 1.0. Best for high-quality data."
        )
        if st.button("Select: Deterministic", use_container_width=True, type="primary"):
            st.session_state["linkage_type"] = "deterministic"
            _go_to(4)

    with c2:
        st.subheader("Probabilistic")
        st.write(
            "Fellegi-Sunter model trained by EM. Each pair receives a match_probability "
            "0-1 based on field-level agreement. Handles typos and missing values. "
            "Takes 1-2 minutes for training."
        )
        if st.button("Select: Probabilistic", use_container_width=True, type="primary"):
            st.session_state["linkage_type"] = "probabilistic"
            _go_to(4)