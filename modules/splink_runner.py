# =============================================================================
# modules/splink_runner.py
# PURPOSE: Wrap Splink's linkage and deduplication workflow.
#          Mirrors logic from:
#            - linkage_workflow/templates/1_train_model_deterministic.ipynb
#            - linkage_workflow/templates/1_train_model_probabilistic.ipynb
#          Enhanced to extract model parameters, missingness stats, and
#          blocking-rule comparison counts for the SeRP-style PDF report.
# =============================================================================

import io
import math
import multiprocessing
import tempfile
from typing import Optional

import duckdb
import pandas as pd

from splink import DuckDBAPI, Linker
import splink.comparison_library as cl
import splink.blocking_rule_library as brl

# ─────────────────────────────────────────────────────────────────────────────
# FIELD → COMPARISON mapping
# Maps each dataset column to an appropriate Splink comparison strategy.
# NameComparison uses Jaro-Winkler fuzzy matching (good for typos in names).
# DateOfBirthComparison handles transpositions and date-range differences.
# ExactMatch is used for categorical fields (city, email, gender, postcode).
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_COMPARISONS = {
    "first_name": lambda: cl.NameComparison("first_name"),
    "surname":    lambda: cl.NameComparison("surname"),
    "dob":        lambda: cl.DateOfBirthComparison("dob", input_is_string=True),
    "city":       lambda: cl.ExactMatch("city"),
    "email":      lambda: cl.ExactMatch("email"),
    "gender":     lambda: cl.ExactMatch("gender"),
    "postcode":   lambda: cl.ExactMatch("postcode"),
}

# ─────────────────────────────────────────────────────────────────────────────
# FIELD → BLOCKING RULE mapping
# Single-field blocking rules only.  Multi-field rules cause Splink 4.0.x to
# create SaltedBlockingRules that are incompatible with u-probability sampling
# on single-CPU environments.
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_BLOCKING_RULES = {
    "first_name": lambda: brl.block_on("first_name"),
    "surname":    lambda: brl.block_on("surname"),
    "dob":        lambda: brl.block_on("dob"),
    "city":       lambda: brl.block_on("city"),
    "email":      lambda: brl.block_on("email"),
    "gender":     lambda: brl.block_on("gender"),
    "postcode":   lambda: brl.block_on("postcode"),
}

DEFAULT_CLUSTER_THRESHOLD     = 0.8    # Cluster together records above this probability
DEFAULT_MATCH_WEIGHT_THRESHOLD = -5.0  # Accept most edges; clustering threshold filters


# =============================================================================
# ── DATA EXTRACTION HELPERS ──────────────────────────────────────────────────
# These are called inside run_linkage() to capture extra data for the PDF report.
# =============================================================================

def _compute_missingness(df: pd.DataFrame, fields: list) -> dict:
    """Compute per-field completeness (% non-null values) for each linkage field.
    Returns {field_name: pct_complete} where pct_complete is 0-100.
    Used in the Datasets section of the SeRP-style PDF report."""
    return {
        field: round(df[field].notna().mean() * 100, 1)  # Percentage complete
        for field in fields
        if field in df.columns    # Only include fields actually present in the DataFrame
    }


def _extract_model_params(linker: Linker) -> dict:
    """Extract trained m/u probabilities and match weights from the Splink linker.

    Called after EM training; returns a structured dict used to plot the
    Match Weights chart and Parameter Estimates chart in the PDF report.
    Returns an empty dict on any access error (deterministic mode is fine).

    Structure returned:
      {
        "comparisons": [
          {
            "field": "first_name",
            "levels": [
              {"label": "Exact match", "m_prob": 0.9, "u_prob": 0.01,
               "match_weight": 6.49, "is_null": False},
              ...
            ]
          },
          ...
        ],
        "prior_log_odds": -10.2,      # log2(lambda / (1-lambda))
        "training_complete": True,
      }
    """
    params = {
        "comparisons":       [],     # One entry per comparison field
        "prior_log_odds":    None,   # Starting match weight (prior)
        "training_complete": False,  # Flag: True only if extraction succeeded
    }

    try:
        settings = linker._settings_obj     # Splink 4 internal settings object

        # ── Extract prior match probability (lambda) ─────────────────────────
        try:
            lam = settings._probability_two_random_records_match  # P(match)
            if lam and 0 < lam < 1:
                params["prior_log_odds"] = math.log2(lam / (1.0 - lam))
            else:
                params["prior_log_odds"] = -10.0          # Safe fallback
        except Exception:
            params["prior_log_odds"] = None

        # ── Extract per-level m/u probabilities for every comparison ─────────
        for comp in settings.comparisons:
            comp_info = {
                "field":  comp._output_column_name,   # e.g. "first_name"
                "levels": [],                          # One dict per comparison level
            }
            for level in comp.comparison_levels:
                m     = getattr(level, "m_probability", None)   # P(agree | match)
                u     = getattr(level, "u_probability", None)   # P(agree | non-match)
                label = getattr(level, "label_for_charts", "Unknown level")
                null  = getattr(level, "_is_null_level", False) # True for null levels

                # Compute match weight = log2(m/u); skip null levels and zeros
                if m and u and u > 0 and not null:
                    weight = math.log2(m / u)
                else:
                    weight = None

                comp_info["levels"].append({
                    "label":        label,
                    "m_prob":       m,
                    "u_prob":       u,
                    "match_weight": weight,
                    "is_null":      null,
                })
            params["comparisons"].append(comp_info)

        params["training_complete"] = True    # Only set True on full success
    except Exception:
        pass    # Return partial dict; caller must guard on training_complete flag

    return params


def _extract_blocking_counts(df_predict: pd.DataFrame, blocking_rule_sqls: list) -> list:
    """Count pairwise comparisons generated by each blocking rule.

    Splink 4 adds a 'match_key' integer column to df_predict indicating which
    blocking rule (0-indexed) produced each candidate pair.

    Returns a list of dicts: [{"rule_index": 0, "rule_sql": "...", "n": 1234}, ...]
    Sorted by rule_index.  Returns empty list if match_key column is absent.
    """
    if "match_key" not in df_predict.columns:
        return []    # match_key not available (deterministic link may not include it)

    try:
        con = duckdb.connect()    # Temporary in-memory DuckDB connection
        con.register("df_predict", df_predict)

        # Count how many predictions each blocking rule contributed
        counts_df = con.sql("""
            SELECT CAST(match_key AS INTEGER) AS rule_index,
                   COUNT(*) AS n
            FROM df_predict
            GROUP BY rule_index
            ORDER BY rule_index
        """).df()
        con.close()

        results = []
        for _, row in counts_df.iterrows():
            idx = int(row["rule_index"])
            # Map rule index to its SQL string; fallback if index out of range
            sql = blocking_rule_sqls[idx] if idx < len(blocking_rule_sqls) else f"Rule {idx}"
            results.append({
                "rule_index": idx,
                "rule_sql":   sql,           # SQL string for the blocking rule
                "n":          int(row["n"]), # Number of comparisons from this rule
            })
        return results
    except Exception:
        return []    # Never crash; blocking counts are supplementary data


def _compute_unlinkables(df_predict: pd.DataFrame, n_records: int) -> tuple:
    """Compute the 'unlinkable records' curve (from SeRP Edge Metrics section).

    For each match-weight threshold t, the curve shows what percentage of
    input records have NO predicted edge with match_weight >= t.  A high
    unlinkable percentage at a given threshold means many records cannot
    be matched with that confidence level.

    Returns (thresholds, unlinkable_pcts) as paired lists.
    """
    if "match_weight" not in df_predict.columns or n_records == 0:
        return [], []

    # Sample thresholds from -20 to +20 in 0.5-unit steps
    thresholds = [t * 0.5 for t in range(-40, 41)]  # -20 to +20 step 0.5
    unlinkable_pcts = []

    try:
        con = duckdb.connect()
        con.register("df_predict", df_predict)

        for t in thresholds:
            # Count unique IDs (left-side) with at least one edge at this threshold
            result = con.sql(f"""
                SELECT COUNT(DISTINCT unique_id_l) AS n_linked
                FROM df_predict
                WHERE match_weight >= {t}
            """).fetchone()
            n_linked = result[0] if result else 0
            # Unlinkable = records with NO edge at or above threshold
            pct = max(0.0, (n_records - n_linked) / n_records * 100.0)
            unlinkable_pcts.append(round(pct, 1))

        con.close()
    except Exception:
        return [], []

    return thresholds, unlinkable_pcts


# =============================================================================
# ── CORE SPLINK WORKFLOW FUNCTIONS ───────────────────────────────────────────
# =============================================================================

def _build_comparisons(selected_fields: list, comp_types: dict = None) -> list:
    """Return Splink comparison objects for selected fields.

    comp_types: optional dict mapping field_name → comparison type string.
    Supported strings: NameComparison, DateOfBirthComparison, ExactMatch,
    LevenshteinAtThresholds, JaroWinklerAtThresholds, EmailComparison,
    PostcodeComparison.
    Falls back to _FIELD_COMPARISONS for known fake1000 fields, then ExactMatch.
    """
    comps = []
    for f in selected_fields:
        # User-specified comparison type takes priority (upload flow)
        if comp_types and f in comp_types:
            ct = comp_types[f]
            if ct == "NameComparison":
                comps.append(cl.NameComparison(f))
            elif ct == "DateOfBirthComparison":
                comps.append(cl.DateOfBirthComparison(f, input_is_string=True))
            elif ct == "LevenshteinAtThresholds":
                comps.append(cl.LevenshteinAtThresholds(f, [1, 2]))
            elif ct == "JaroWinklerAtThresholds":
                comps.append(cl.JaroWinklerAtThresholds(f, [0.9, 0.7]))
            elif ct == "EmailComparison":
                comps.append(cl.EmailComparison(f))
            elif ct == "PostcodeComparison":
                comps.append(cl.PostcodeComparison(f))
            else:
                comps.append(cl.ExactMatch(f))   # Default for any unknown type
        elif f in _FIELD_COMPARISONS:
            comps.append(_FIELD_COMPARISONS[f]()) # Known fake1000 fields
        else:
            comps.append(cl.ExactMatch(f))        # Safe fallback for any other field
    return comps


def _build_blocking_rules(blocking_toggles: dict) -> list:
    """Return active Splink blocking rule objects for any field name.

    Supports three rule types:
      - Single field:    "first_name"          → brl.block_on("first_name")
      - Composite field: "first_name+surname"  → brl.block_on("first_name","surname")
      - Any field not in _FIELD_BLOCKING_RULES uses generic brl.block_on(key)
        so uploaded datasets with arbitrary column names work correctly.
    """
    active = []
    for key, enabled in blocking_toggles.items():
        if not enabled:
            continue
        if "+" in key:
            # Composite rule: "field1+field2" → brl.block_on("field1","field2")
            fields = [f.strip() for f in key.split("+") if f.strip()]
            if len(fields) >= 2:
                active.append(brl.block_on(*fields))
        else:
            # Single field: use brl.block_on for ANY column name
            active.append(brl.block_on(key))
    if not active:
        raise ValueError("At least one blocking rule must be enabled.")
    return active


def _validate_and_filter_settings(settings: dict, input_tables: list) -> dict:
    """Remove comparisons and blocking rules for columns absent from any input table.

    This is critical for link_only mode where Dataset A and Dataset B may have
    different schemas (e.g. NC voter registration vs voter history).
    If a blocking rule references a column that doesn't exist in one table,
    DuckDB raises a Binder Error at prediction time.

    Strategy:
      1. Find the intersection of columns present in ALL input tables.
      2. Drop any comparison whose output_column_name is not in common_cols.
      3. Drop any blocking rule that references a column not in common_cols.
      4. Raise ValueError if no blocking rules survive (nothing to link on).
    """
    import re as _re

    # Compute the set of columns that exist in every input table
    common_cols = set(input_tables[0].columns)
    for df in input_tables[1:]:
        common_cols &= set(df.columns)

    # Filter comparisons: keep only those whose column exists in all tables
    original_comps = settings.get("comparisons", [])
    settings["comparisons"] = [
        c for c in original_comps
        if c.get("output_column_name", "") in common_cols
    ]
    dropped_comps = len(original_comps) - len(settings["comparisons"])
    if dropped_comps:
        import warnings
        warnings.warn(
            f"{dropped_comps} comparison(s) dropped: column(s) not present in all datasets."
        )

    # Filter blocking rules: parse column names from SQL and check all exist
    original_rules = settings.get("blocking_rules_to_generate_predictions", [])
    valid_rules = []
    for rule in original_rules:
        sql = rule.get("blocking_rule", "") if isinstance(rule, dict) else str(rule)
        # Extract all l."col" column references from the SQL string
        referenced_cols = _re.findall(r'l\."([^"]+)"', sql)
        if all(c in common_cols for c in referenced_cols):
            valid_rules.append(rule)

    settings["blocking_rules_to_generate_predictions"] = valid_rules

    if not valid_rules:
        common_sorted = sorted(common_cols - {"source_dataset"})
        raise ValueError(
            "No valid blocking rules remain after column validation. "
            "All blocking fields must exist in BOTH Dataset A and Dataset B. "
            f"Columns present in both datasets: {common_sorted}. "
            "Please reconfigure blocking rules to use only these columns."
        )
    return settings


def _build_model_settings(link_type, selected_fields, blocking_toggles,
                          comp_types: dict = None) -> dict:
    """Assemble the Splink settings dict from user inputs.
    comp_types: optional dict mapping field → comparison type string (upload flow).
    Converts comparison objects and blocking rules to dicts for Splink 4.x."""
    comparisons    = _build_comparisons(selected_fields, comp_types)
    blocking_rules = _build_blocking_rules(blocking_toggles)
    return {
        "link_type":         link_type,
        "unique_id_column_name": "unique_id",
        "comparisons": [c.create_comparison_dict("duckdb") for c in comparisons],
        "blocking_rules_to_generate_predictions": [
            r.create_blocking_rule_dict("duckdb") for r in blocking_rules
        ],
        "retain_matching_columns":                True,
        "retain_intermediate_calculation_columns": True,
        "max_iterations":  25,
        "em_convergence":  0.0001,
    }


def _train_probabilistic(linker: Linker, selected_fields: list) -> None:
    """Three-step EM training for a probabilistic model.

    Uses single-field blocking rules throughout to avoid Splink 4.0.x's
    SaltedBlockingRule incompatibility on single-CPU machines.
    The cpu_count monkeypatch forces Splink to use 2 salting partitions
    (required minimum) during the u-probability random-sampling step.
    """
    PRIORITY  = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]
    available = [f for f in PRIORITY if f in selected_fields]
    primary   = available[0] if available else selected_fields[0]
    secondary = available[1] if len(available) > 1 else None

    # Step 1: Prior estimate
    linker.training.estimate_probability_two_random_records_match(
        [brl.block_on(primary)], recall=0.6
    )

    # Step 2: u-probabilities via random sampling (with cpu_count patch)
    _orig = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: 2    # Force >= 2 salting partitions
    try:
        linker.training.estimate_u_using_random_sampling(1e5)
    finally:
        multiprocessing.cpu_count = _orig    # Always restore

    # Step 3: EM training for m-probabilities
    linker.training.estimate_parameters_using_expectation_maximisation(
        brl.block_on(primary), fix_u_probabilities=True
    )
    if secondary:
        linker.training.estimate_parameters_using_expectation_maximisation(
            brl.block_on(secondary), fix_u_probabilities=True
        )


def _render_cluster_studio_html(linker, df_predict, df_cluster) -> str:
    """Generate Splink cluster studio HTML for embedding in Streamlit.
    Returns empty string if generation fails (never crashes the app)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name

        linker.visualisations.cluster_studio_dashboard(
            df_predict=df_predict,
            df_clustered=df_cluster,
            out_path=tmp_path,
            overwrite=True,
            return_html_as_string=True,
        )

        import os
        if os.path.exists(tmp_path):
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_str = f.read()
            os.remove(tmp_path)
            return html_str
    except Exception:
        pass
    return ""


# =============================================================================
# ── PUBLIC API ────────────────────────────────────────────────────────────────
# =============================================================================

def run_linkage_from_json(
    model_json:       dict,
    fakea:            pd.DataFrame,
    fakeb:            Optional[pd.DataFrame],
    operation_mode:   str,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict:
    """Run prediction from a pre-trained Splink model JSON (advanced flow).

    Accepts any valid Splink 4.x settings JSON that already contains trained
    m/u probabilities.  No EM training is performed here; the model is used
    as-is for inference.

    Args:
        model_json       : Parsed Splink settings dict (from uploaded .json file)
        fakea            : Dataset A (source_dataset = 'A')
        fakeb            : Dataset B or None
        operation_mode   : 'dedupe' or 'link_dedupe'
        cluster_threshold: Match probability threshold for clustering

    Returns same dict structure as run_linkage() so the rest of the app
    (analysis, PDF, confusion matrix) works identically for both flows.
    """
    # Work on a copy so we never mutate the user's uploaded dict
    settings = dict(model_json)

    # Force column retention so metrics and explorer can inspect field values
    settings["retain_intermediate_calculation_columns"] = True
    settings["retain_matching_columns"]                 = True

    # Override link_type if the JSON doesn't match the chosen operation mode
    if operation_mode == "dedupe":
        settings["link_type"] = "dedupe_only"
        df_input   = fakea.copy()
        df_input["source_dataset"] = "A"
        input_tables    = [df_input]
        n_input_records = len(df_input)
    else:
        settings["link_type"] = "link_only"
        input_tables    = [fakea, fakeb]
        n_input_records = len(fakea) + (len(fakeb) if fakeb is not None else 0)

    # Pull the field names used in comparisons for missingness reporting
    comparison_fields = [
        c.get("output_column_name", "")
        for c in settings.get("comparisons", [])
        if c.get("output_column_name")
    ]

    # Compute missingness before building the linker
    missingness_a = _compute_missingness(fakea, comparison_fields)
    missingness_b = (
        _compute_missingness(fakeb, comparison_fields)
        if fakeb is not None and operation_mode != "dedupe"
        else {}
    )

    # Build linker from uploaded settings – no training step
    db_api = DuckDBAPI()
    linker  = Linker(
        input_table_or_tables=input_tables,
        settings=settings,
        db_api=db_api,
        set_up_basic_logging=False,
    )

    # Run prediction (threshold very low so all pairs are returned)
    df_predict = linker.inference.predict(
        threshold_match_weight=DEFAULT_MATCH_WEIGHT_THRESHOLD
    )

    # Cluster
    df_cluster = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict,
        threshold_match_probability=cluster_threshold,
    )

    df_predict_pd = df_predict.as_pandas_dataframe()
    df_cluster_pd = df_cluster.as_pandas_dataframe()

    # Extract trained model parameters for the PDF match weights chart
    model_params = _extract_model_params(linker)

    # Extract blocking SQL strings from the settings
    blocking_rule_sqls = [
        (r.get("blocking_rule", "") if isinstance(r, dict) else str(r))
        for r in settings.get("blocking_rules_to_generate_predictions", [])
    ]
    blocking_counts = _extract_blocking_counts(df_predict_pd, blocking_rule_sqls)

    # Unlinkable records curve
    thresh, pcts = _compute_unlinkables(df_predict_pd, n_input_records)

    # Cluster studio HTML
    cluster_html = _render_cluster_studio_html(linker, df_predict, df_cluster)

    # Build a run_config that the rest of the app can consume
    # Parse blocking toggles: each SQL like 'l."field" = r."field"' → field name
    blocking_toggles_from_json = {}
    for sql in blocking_rule_sqls:
        # Extract the field name from 'l."field" = r."field"' pattern
        import re
        matches = re.findall(r'l\."([^"]+)"', sql)
        if matches:
            blocking_toggles_from_json[matches[0]] = True

    run_config = {
        "operation_mode":    operation_mode,
        "linkage_type":      "probabilistic",  # JSON models are trained → probabilistic
        "selected_fields":   comparison_fields,
        "blocking_toggles":  blocking_toggles_from_json,
        "cluster_threshold": cluster_threshold,
        "link_type":         settings.get("link_type", "dedupe_only"),
        "from_json":         True,             # Flag so UI can show "Advanced flow"
    }

    return {
        "df_predict":       df_predict_pd,
        "df_cluster":       df_cluster_pd,
        "cluster_html":     cluster_html,
        "n_edges":          len(df_predict_pd),
        "n_clusters":       df_cluster_pd["cluster_id"].nunique(),
        "n_input_records":  n_input_records,
        "settings_used":    settings,
        "model_params":     model_params,
        "missingness_a":    missingness_a,
        "missingness_b":    missingness_b,
        "blocking_counts":  blocking_counts,
        "unlinkables":      {"thresholds": thresh, "pcts": pcts},
        "run_config":       run_config,
    }


# =============================================================================
# ── INTERACTIVE BLOCKING EXPLORER ────────────────────────────────────────────
# Lets users toggle blocking rules on/off and see df_predict update live.
# Uses retain_matching_columns=True so field values are already in df_predict,
# meaning no extra join to the original datasets is needed.
# =============================================================================

def build_coverage_matrix(
    df_predict:     pd.DataFrame,
    active_fields:  list,
) -> pd.DataFrame:
    """Compute which blocking rules would cover each pair in df_predict.

    Because retain_matching_columns=True, df_predict already contains
    field_l and field_r columns for every comparison field.  A blocking rule
    for field X covers a pair if field_X_l == field_X_r (exact match).

    Returns a slim DataFrame with:
      unique_id_l, unique_id_r, source_dataset_l, source_dataset_r,
      match_key, match_probability, match_weight,
      covers_<field>  (bool)  for each active field
    """
    # Start with the pair identity and score columns only (keeps it lightweight)
    id_cols    = ["unique_id_l", "unique_id_r", "source_dataset_l", "source_dataset_r"]
    score_cols = ["match_key", "match_probability", "match_weight"]
    keep       = [c for c in id_cols + score_cols if c in df_predict.columns]
    result     = df_predict[keep].copy()

    for field in active_fields:
        col_l = f"{field}_l"
        col_r = f"{field}_r"
        if col_l in df_predict.columns and col_r in df_predict.columns:
            # True when both sides have a non-null, identical value for this field
            result[f"covers_{field}"] = (
                df_predict[col_l].notna()
                & df_predict[col_r].notna()
                & (df_predict[col_l].astype(str) == df_predict[col_r].astype(str))
            )
        else:
            # Field columns not retained → assume not covered
            result[f"covers_{field}"] = False

    return result


def filter_predict_by_active_rules(
    df_predict:       pd.DataFrame,
    coverage_matrix:  pd.DataFrame,
    active_toggles:   dict,
) -> pd.DataFrame:
    """Filter df_predict to pairs covered by at least one active blocking rule.

    If a pair was originally captured by rule A (now disabled) but would also
    be captured by rule B (still active), the pair is retained.  The coverage
    matrix encodes all rules that WOULD cover each pair, not just the one
    that originally generated it (match_key).

    Returns a filtered df_predict with a new 'effective_rule' column showing
    the name of the first active rule that covers each pair.
    """
    active_fields = [f for f, v in active_toggles.items() if v]
    if not active_fields:
        # No active rules → empty table
        return df_predict.iloc[0:0].copy()

    cover_cols = [f"covers_{f}" for f in active_fields
                  if f"covers_{f}" in coverage_matrix.columns]
    if not cover_cols:
        return df_predict.copy()

    # A pair is included if ANY active coverage column is True
    mask = coverage_matrix[cover_cols].any(axis=1)

    # Get the pair IDs that survive the filter
    id_cols   = ["unique_id_l", "unique_id_r", "source_dataset_l", "source_dataset_r"]
    id_cols   = [c for c in id_cols if c in coverage_matrix.columns]
    surviving = coverage_matrix[mask][id_cols].copy()

    # Add 'effective_rule': the name of the first active rule covering the pair
    first_rule = []
    for _, row in coverage_matrix[mask].iterrows():
        rule_name = "unknown"
        for f in active_fields:
            if row.get(f"covers_{f}", False):
                rule_name = f
                break
        first_rule.append(rule_name)
    surviving["effective_rule"] = first_rule

    # Join back to get all original df_predict columns for surviving pairs
    merge_keys = [c for c in id_cols if c in df_predict.columns]
    filtered = df_predict.merge(surviving, on=merge_keys, how="inner")

    return filtered


def recluster_filtered(
    df_predict_filtered: pd.DataFrame,
    fakea:               pd.DataFrame,
    fakeb:               Optional[pd.DataFrame],
    threshold:           float = DEFAULT_CLUSTER_THRESHOLD,
) -> pd.DataFrame:
    """Re-cluster a filtered df_predict using Splink's standalone clustering.

    Does not require a Linker instance; uses the standalone function from
    splink.clustering which runs connected components on the edge list.
    This is fast even for thousands of pairs since the graph is small.

    Returns a df_cluster DataFrame (unique_id, cluster_id, source_dataset).
    Returns empty DataFrame if clustering fails.
    """
    try:
        # Standalone clustering function (no Linker required)
        from splink.clustering import cluster_pairwise_predictions_at_threshold as _cluster

        # Build the nodes table from original datasets
        if fakeb is not None:
            nodes = pd.concat([fakea, fakeb], ignore_index=True)
        else:
            nodes = fakea.copy()

        db_api = DuckDBAPI()    # Fresh in-memory DuckDB for this operation
        result = _cluster(
            nodes=nodes,                        # All records (nodes in the graph)
            edges=df_predict_filtered,           # Filtered edge list
            db_api=db_api,
            node_id_column_name="unique_id",     # Column that uniquely identifies records
            threshold_match_probability=threshold,
        )
        return result.as_pandas_dataframe()
    except Exception as e:
        return pd.DataFrame()                    # Return empty on any error; never crash


# =============================================================================
# ── HYPERPARAMETER-AWARE TRAINING ─────────────────────────────────────────────
# Updated run_linkage signature: accepts hyperparams dict so the UI can
# expose EM iterations, convergence, recall estimate, and sample size.
# =============================================================================

def run_linkage(
    fakea:            pd.DataFrame,
    fakeb:            Optional[pd.DataFrame],
    selected_fields:  list,
    blocking_toggles: dict,
    operation_mode:   str,
    linkage_type:     str,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    hyperparams:      Optional[dict] = None,   # EM training hyperparameters
    composite_rules:  Optional[dict] = None,   # composite blocking rules e.g. "first_name+dob"
    comp_types:       Optional[dict] = None,   # field → comparison type (upload flow)
) -> dict:
    """Thin wrapper: merges composite rules into blocking_toggles, then delegates
    to the internal logic.  All hyperparams are forwarded to the model settings
    and training functions.

    hyperparams keys (all optional, defaults in parentheses):
      max_iterations  : int   (25)      - max EM iterations
      em_convergence  : float (0.0001)  - stop when change < this
      recall_estimate : float (0.6)     - used in prior probability estimate
    """
    hp = hyperparams or {}

    # Merge composite rules (e.g. "first_name+surname") into blocking_toggles
    merged_toggles = dict(blocking_toggles)
    for key, enabled in (composite_rules or {}).items():
        merged_toggles[key] = enabled

    # ── Determine Splink link_type ─────────────────────────────────────────────
    link_type = "dedupe_only" if operation_mode == "dedupe" else "link_only"

    # ── Prepare input tables ───────────────────────────────────────────────────
    if operation_mode == "dedupe":
        df_for_dedupe = fakea.copy()
        df_for_dedupe["source_dataset"] = "A"
        input_tables    = [df_for_dedupe]
        n_input_records = len(df_for_dedupe)
    else:
        input_tables    = [fakea, fakeb]
        n_input_records = len(fakea) + len(fakeb)

    # ── Missingness ────────────────────────────────────────────────────────────
    missingness_a = _compute_missingness(fakea, selected_fields)
    missingness_b = (
        _compute_missingness(fakeb, selected_fields)
        if fakeb is not None and operation_mode != "dedupe"
        else {}
    )

    # ── Build settings (now uses hyperparams for EM config) ───────────────────
    settings = _build_model_settings_hp(
        link_type, selected_fields, merged_toggles, hp,
        comp_types=comp_types,    # Pass user-specified comparison types
    )

    # ── Ensure Splink-required columns exist in EVERY input table ────────────
    # unique_id and source_dataset MUST be present in all tables before the
    # Linker is created.  If either is absent from any table we add it now,
    # so the column intersection used by the UNION ALL alignment below always
    # includes these two columns.
    import re as _re_sr
    for _i, _df in enumerate(input_tables):
        _df = _df.copy()
        if "unique_id" not in _df.columns:
            _prefix = ("AB"[_i] if _i < 2 else str(_i))
            _df.insert(0, "unique_id",
                       _prefix + "_" + pd.Series(range(len(_df))).astype(str))
        if "source_dataset" not in _df.columns:
            _df["source_dataset"] = "A" if _i == 0 else "B"
        input_tables[_i] = _df

    # ── Column alignment for link mode ────────────────────────────────────────
    # Splink generates a UNION ALL of all input tables to create an internal
    # concatenated table used for all subsequent SQL. UNION ALL requires
    # IDENTICAL column lists in every table. If Dataset A has columns that
    # Dataset B lacks (e.g. NC voter registration has first_name but voter
    # history does not), the UNION ALL SQL fails with "column not found".
    # Fix: restrict ALL input tables to the intersection of their columns
    # BEFORE passing them to the Linker. Comparisons and blocking rules are
    # then validated against this common schema by _validate_and_filter_settings.
    if len(input_tables) > 1:
        # Compute the set of columns that exist in every input table
        _common_schema = set(input_tables[0].columns)
        for _df in input_tables[1:]:
            _common_schema &= set(_df.columns)
        # Drop columns that are not shared so the UNION ALL schema is consistent
        input_tables = [
            _df[[c for c in _df.columns if c in _common_schema]].copy()
            for _df in input_tables
        ]

    # Remove comparisons and blocking rules whose columns are not in all tables.
    settings = _validate_and_filter_settings(settings, input_tables)

    db_api = DuckDBAPI()
    linker  = Linker(
        input_table_or_tables=input_tables,
        settings=settings,
        db_api=db_api,
        set_up_basic_logging=False,
    )

    # ── Run model ─────────────────────────────────────────────────────────────
    model_params = {}
    if linkage_type == "deterministic":
        df_predict_raw    = linker.inference.deterministic_link()
        df_predict_pd_raw = df_predict_raw.as_pandas_dataframe()
        df_predict_pd_raw["match_probability"] = 1.0
        df_predict_pd_raw["match_weight"]      = 100.0
        if "source_dataset_l" not in df_predict_pd_raw.columns:
            df_predict_pd_raw["source_dataset_l"] = "A"
        if "source_dataset_r" not in df_predict_pd_raw.columns:
            df_predict_pd_raw["source_dataset_r"] = "A"
        df_predict = linker.table_management.register_table(
            df_predict_pd_raw, "df_predict_enriched"
        )
    else:
        # Probabilistic with user-supplied hyperparams
        _train_probabilistic_hp(linker, selected_fields, hp)
        model_params = _extract_model_params(linker)
        df_predict   = linker.inference.predict(
            threshold_match_weight=DEFAULT_MATCH_WEIGHT_THRESHOLD
        )

    # ── Cluster ────────────────────────────────────────────────────────────────
    df_cluster = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict,
        threshold_match_probability=cluster_threshold,
    )

    df_predict_pd = df_predict.as_pandas_dataframe()
    df_cluster_pd = df_cluster.as_pandas_dataframe()

    # ── Blocking counts, unlinkables, cluster studio ──────────────────────────
    blocking_rule_sqls = [
        r["blocking_rule"]
        for r in settings["blocking_rules_to_generate_predictions"]
    ]
    blocking_counts = _extract_blocking_counts(df_predict_pd, blocking_rule_sqls)
    thresh, pcts    = _compute_unlinkables(df_predict_pd, n_input_records)
    cluster_html    = _render_cluster_studio_html(linker, df_predict, df_cluster)

    return {
        "df_predict":       df_predict_pd,
        "df_cluster":       df_cluster_pd,
        "cluster_html":     cluster_html,
        "n_edges":          len(df_predict_pd),
        "n_clusters":       df_cluster_pd["cluster_id"].nunique(),
        "n_input_records":  n_input_records,
        "settings_used":    settings,
        "model_params":     model_params,
        "missingness_a":    missingness_a,
        "missingness_b":    missingness_b,
        "blocking_counts":  blocking_counts,
        "unlinkables":      {"thresholds": thresh, "pcts": pcts},
        "run_config": {
            "operation_mode":    operation_mode,
            "linkage_type":      linkage_type,
            "selected_fields":   selected_fields,
            "blocking_toggles":  merged_toggles,
            "cluster_threshold": cluster_threshold,
            "link_type":         link_type,
            "hyperparams":       hp,
            "from_json":         False,
        },
    }


def _build_model_settings_hp(link_type, selected_fields, blocking_toggles, hp,
                              comp_types: dict = None) -> dict:
    """Build Splink settings dict accepting hyperparams and comparison types."""
    comparisons    = _build_comparisons(selected_fields, comp_types)  # user types
    blocking_rules = _build_blocking_rules(blocking_toggles)          # handles composites
    return {
        "link_type":         link_type,
        "unique_id_column_name": "unique_id",
        "comparisons": [c.create_comparison_dict("duckdb") for c in comparisons],
        "blocking_rules_to_generate_predictions": [
            r.create_blocking_rule_dict("duckdb") for r in blocking_rules
        ],
        "retain_matching_columns":                True,
        "retain_intermediate_calculation_columns": True,
        "max_iterations":  hp.get("max_iterations", 25),      # exposed to UI
        "em_convergence":  hp.get("em_convergence", 0.0001),  # exposed to UI
    }


def _train_probabilistic_hp(linker, selected_fields, hp) -> None:
    """Train probabilistic model using user-supplied hyperparameters."""
    recall = hp.get("recall_estimate", 0.6)      # User-adjustable recall for prior

    PRIORITY  = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]
    available = [f for f in PRIORITY if f in selected_fields]
    primary   = available[0] if available else selected_fields[0]
    secondary = available[1] if len(available) > 1 else None

    # Step 1: Prior using user-specified recall estimate
    linker.training.estimate_probability_two_random_records_match(
        [brl.block_on(primary)], recall=recall
    )

    # Step 2: u-probabilities (cpu_count patch for single-CPU environments)
    _orig = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: 2
    try:
        linker.training.estimate_u_using_random_sampling(1e5)
    finally:
        multiprocessing.cpu_count = _orig

    # Step 3: EM training
    linker.training.estimate_parameters_using_expectation_maximisation(
        brl.block_on(primary), fix_u_probabilities=True
    )
    if secondary:
        linker.training.estimate_parameters_using_expectation_maximisation(
            brl.block_on(secondary), fix_u_probabilities=True
        )


# (Stale _build_blocking_rules removed: now handled by the generic version above which uses brl.block_on(key) for ANY field name)


# =============================================================================
# ── SAVE MODEL AS JSON ────────────────────────────────────────────────────────
# Reconstruct a full Splink model JSON from settings + extracted model params.
# The output is accepted by run_linkage_from_json() and the advanced flow.
# =============================================================================

def reconstruct_model_json(settings_used: dict, model_params: dict) -> dict:
    """Build a Splink model JSON from settings_used + trained model_params.

    Takes the settings dict returned by run_linkage() and the model_params
    dict from _extract_model_params(), and injects the trained m/u probabilities
    back into the comparison levels so the JSON can be used to skip training.

    Works for both probabilistic (m/u populated) and deterministic runs
    (m/u will be null in the output, which is valid for deterministic replay).

    Returns a dict that json.dumps() can serialise directly.
    """
    import copy
    import math as _math

    model = copy.deepcopy(settings_used)    # Never mutate the original

    # ── Inject prior probability ───────────────────────────────────────────────
    prior_log_odds = model_params.get("prior_log_odds")
    if prior_log_odds is not None:
        try:
            # Convert log-odds back to probability: P = 2^W / (1 + 2^W)
            prob = 2 ** prior_log_odds / (1 + 2 ** prior_log_odds)
            model["probability_two_random_records_match"] = round(prob, 8)
        except Exception:
            pass

    # ── Inject m/u probabilities into each comparison level ──────────────────
    # Build a field → levels list lookup from model_params
    params_by_field = {
        comp["field"]: comp.get("levels", [])
        for comp in model_params.get("comparisons", [])
        if "field" in comp
    }

    for comp_dict in model.get("comparisons", []):
        field       = comp_dict.get("output_column_name", "")
        lvl_params  = params_by_field.get(field, [])    # May be empty for deterministic

        for j, lvl in enumerate(comp_dict.get("comparison_levels", [])):
            if j < len(lvl_params):
                lp = lvl_params[j]
                # Only inject if the value is not None (null levels stay null)
                if lp.get("m_prob") is not None:
                    lvl["m_probability"] = lp["m_prob"]
                if lp.get("u_prob") is not None:
                    lvl["u_probability"] = lp["u_prob"]

    return model
