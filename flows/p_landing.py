# pages/p_landing.py
# Landing page — three mode cards: Standard, Upload Data, Advanced (JSON).

import streamlit as st
from modules.data_builder import build_datasets, get_library_status
from utils.nav import _go_to
from utils.state import clear_run_results


def page_landing() -> None:
    st.title("Splink Cohort Builder")
    st.write(
        "Choose how you want to work. Standard mode walks you through every "
        "configuration step using the built-in fake1000 dataset. Upload mode "
        "lets you bring your own CSV or TXT files with a full EDA cleaning "
        "pipeline. Advanced mode accepts a pre-trained Splink model JSON and "
        "jumps straight to prediction and analysis."
    )
    st.divider()

    col_std, col_up, col_adv = st.columns(3, gap="large")

    # ── Standard mode ─────────────────────────────────────────────────────────
    with col_std:
        st.subheader("Standard Mode")
        st.caption("Guided workflow · built-in dataset")
        st.write(
            "Use the fake1000 dataset (1,000 synthetic UK records with name, "
            "DOB, city, email, gender, and postcode). Guided step-by-step "
            "through field selection, blocking rules, and linkage type."
        )
        if st.button("Use dummy dataset", use_container_width=True, type="primary"):
            with st.spinner("Building fake1000 dataset..."):
                try:
                    _, fakea, fakeb = build_datasets()
                    st.session_state["fakea"]        = fakea
                    # Store fakeb in a staging key: page_operation() will copy
                    # it to "fakeb" only if the user chooses link+deduplicate.
                    # This restores the explicit user choice before committing.
                    st.session_state["std_fakeb"]    = fakeb
                    st.session_state["fakeb"]        = None   # start as None
                    st.session_state["dataset_ready"] = True
                    st.session_state["flow"]         = "standard"
                    libs = get_library_status()
                    if not libs["gender_guesser"]:
                        st.warning("gender-guesser not installed: random gender used. "
                                   "pip install gender-guesser for name inference.")
                    if not libs["pgeocode"]:
                        st.warning("pgeocode not installed: synthetic postcodes used. "
                                   "pip install pgeocode for real UK postcodes.")
                    st.success("Dataset loaded.")
                except Exception as e:
                    st.error(f"Failed to build dataset: {e}")

    # ── Upload mode ───────────────────────────────────────────────────────────
    with col_up:
        st.subheader("Upload Your Data")
        st.caption("CSV or TXT · automated EDA · your fields")
        st.write(
            "Upload one or two CSV/TXT files. The app cleans and standardises "
            "your data (field names, nulls, duplicates, dates) then guides you "
            "through field configuration and blocking rules. Supports URL and "
            "local file path loading for large files."
        )
        if st.button("Upload dataset", use_container_width=True):
            clear_run_results()
            st.session_state["flow"] = "upload"
            _go_to("upload_setup")

    # ── Advanced mode ─────────────────────────────────────────────────────────
    with col_adv:
        st.subheader("Advanced Mode")
        st.caption("Pre-trained model JSON · skip training")
        st.write(
            "Upload a Splink model JSON produced by "
            "linker.misc.save_model_to_json(). Skips all EM training and "
            "jumps straight to prediction, interactive blocking explorer, "
            "and PDF report. Trained models can be saved from the analysis page."
        )
        if st.button("Upload model JSON", use_container_width=True):
            st.session_state["flow"] = "advanced"
            _go_to("advanced_setup")

    # ── Preview if standard dataset is loaded ─────────────────────────────────
    if st.session_state["dataset_ready"] and st.session_state["flow"] == "standard":
        st.divider()
        st.subheader("Dataset A — Preview")
        st.dataframe(st.session_state["fakea"].head(5), use_container_width=True)
        _fb = st.session_state.get("std_fakeb")
        st.caption(
            f"Dataset A: {len(st.session_state['fakea']):,} records"
            + (f"  |  Dataset B available: {len(_fb):,} records (50% sample with controlled errors)"
               if _fb is not None else "")
        )
        st.divider()
        if st.button("Continue to field configuration", type="primary"):
            _go_to(1)

    st.divider()
    i1, i2, i3 = st.columns(3, gap="medium")
    with i1:
        st.markdown("**How cohort building works**")
        st.write(
            "Configure fields, blocking rules, and linkage type. "
            "The model identifies matching records and groups them into entity clusters. "
            "Export the cohort as a CSV with cluster IDs."
        )
    with i2:
        st.markdown("**Linkage and deduplication**")
        st.write(
            "Probabilistic linkage uses Fellegi-Sunter EM training to assign "
            "match probabilities. Deterministic applies exact-match rules. "
            "Both produce entity clusters for cohort building."
        )
    with i3:
        st.markdown("**What you will see**")
        st.write(
            "Match probability distributions, gamma scores, cluster metrics, "
            "Venn diagram, confusion matrix (Precision/Recall/F1/CRL), "
            "interactive blocking explorer, and a downloadable PDF report."
        )
