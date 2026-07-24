# Cohort Builder

A Streamlit application for record linkage and deduplication using Splink and DuckDB. Built at Swansea University as an MVP for cohort construction workflows, targeting both non-technical and technical users.

## Access it online at https://cb-swansea.streamlit.app/


## Three workflows

**Standard mode** — guided seven-step workflow for non-technical users. Loads the built-in fake1000 dataset, walks through field selection, blocking rules, operation mode, and linkage type, then produces analysis and an exportable cohort.  
**Upload mode** — bring your own data. Upload one or two CSV files, run the automated EDA cleaning pipeline, configure fields and blocking rules, then follow the same analysis, comparison, and export steps as standard mode.  
**Advanced mode** — for power users with a pre-trained Splink model. Upload a model JSON file, skip all training, and jump straight to prediction, interactive blocking exploration, and export. Models trained in standard or upload mode can be saved as JSON and reused here.

--- 
**Upload mode** - for users to upload their own datasets, clean and standardise the fields, and then run the analysis. Users can generate an error prone dataset from their original dataset to test out the linkage model.

You can save your exisiting model on Splink using the following code:
```
# Save model to JSON
linker.misc.save_model_to_json("test_splink_model.json", overwrite=True)
```

Before uploading, please check if your file has the following format to ensure consistency between runs and the app accepts the uploaded JSON file
```
{
  "link_type": "dedupe_only",
  "unique_id_column_name": "unique_id",
  "probability_two_random_records_match": 0.000812,
  "comparisons": [ ... ],
  "blocking_rules_to_generate_predictions": [ ... ]
}
```
Please ensure, comparisons contains m and u probabilities. 

---

## Features

- Probabilistic linkage via Expectation-Maximisation (Splink 4.x + DuckDB backend)
- Deterministic linkage with exact-match blocking rules
- Deduplication only, or cross-dataset linkage (Dataset A + Dataset B)
- Three-mode sidebar switcher to move between Standard, Upload, and Advanced flows at any time
- Back navigation with history stack on every page
- Save trained model as JSON for reuse in Advanced mode
- Interactive blocking explorer: toggle rules on/off, live df_predict table update, one-click re-clustering
- Composite blocking rules (e.g. first_name + surname as a single rule)
- Exposed training hyperparameters: EM iterations, convergence threshold, recall estimate
- Confusion matrix with ground truth from the cluster column: TP, FP, FN, Precision, Recall, F1, F*, FDR, FNR
- Precision-Recall curve and CRL (Composite Reliability of Linkage) score
<<<<<<< Updated upstream
- Full metrics suite covering linkage-metrics examples 0–16
- Clickable sidebar navigation with back button and jump-to-export shortcut
- Full metrics suite covering linkage-metrics: match weight histogram, gamma scores, cluster size distribution, confusion matrix, Venn diagram, inter-run edge comparison
- SeRP-style downloadable PDF report with nine sections

---

## EDA pipeline (Upload mode)
When you upload a CSV the following cleaning steps run automatically:

1. Field name standardisation — lowercase, underscores, strip trailing numbers and special characters
2. Field type detection — infers semantic type (first_name, surname, dob, gender, location, postcode, email, id) from column names to drive comparison and blocking suggestions
3. Remove 100%-null columns — columns where every value is missing are dropped
4. Remove 100%-null rows — rows with no values at all are dropped
5. Remove n-1 null rows — rows with only one non-null value are dropped
6. Remove n-2 null rows — rows with only two non-null values are dropped
7. Text cleaning — strip whitespace, Title Case for name fields, lowercase for all other text
8. Duplicate removal — exact duplicate rows are dropped
9. Date standardisation — parses common date formats (DD/MM/YYYY, YYYYMMDD, etc.) and converts to YYYY-MM-DD
10. Correlation check — finds pairs of non-ID columns with >= 95% value-level agreement and asks which field to keep
11. EDA summary display — shows rows removed per step, fields changed, detected types, and a cleaned data preview
12. Download cleaned CSV — the cleaned dataset can be saved before proceeding

### Dataset B options (Upload mode)

- Upload a second CSV directly as Dataset B
- Create a 30% sample of Dataset A with controlled errors introduced (14% name typos, 5% missing DOBs, 15% email variations, 11% city abbreviations, 7% gender errors) for testing linkage
- Deduplication only (no Dataset B required)  
---

## Project structure

```
cohort_builder/
├── app.py                    # Main Streamlit app (three flows, session-state navigation)
├── modules/
│   ├── data_builder.py       # Builds fake1000 with gender and UK postcode
│   ├── eda_engine.py         # Automated EDA and cleaning pipeline for uploaded data
│   ├── splink_runner.py      # Linkage workflow, JSON flow, coverage matrix, re-clustering, model JSON export
│   ├── metrics_engine.py     # All linkage quality metrics (examples 0-16 + confusion matrix)
│   ├── report_gen.py         # SeRP-style PDF report generator
│   └── splink_runner.py 
├── flow/
│   ├── p_advanced.py
│   ├── p_analysis.py
│   ├── p_compare_export.py
│   ├── p_landing.py
│   ├── p_standard.py
│   └── p_upload.py
└── requirements.txt
```

---

## Installation

```
pip install -r requirements.txt
streamlit run app.py
```

Optional dependencies for higher data quality in the generated dataset:

```
pip install gender-guesser pgeocode
```

Both fall back gracefully if not installed.

---

## Core dependencies

streamlit, splink, duckdb, pandas, numpy, plotly, fpdf2, matplotlib

---

## Testing the JSON upload feature
Generate a trained model JSON from any notebook or from the app itself (Save model JSON button on the analysis page), then upload it in Advanced mode. The JSON must be produced by linker.misc.save_model_to_json() or by the app's export function, which injects trained m/u probabilities into the comparison levels.

---

## Datasets

The built-in fake1000 dataset is derived from Splink's fake_1000, augmented with gender (inferred from first_name) and postcode (UK GeoNames lookup by city). Dataset B is a 50% sample of Dataset A with controlled errors: 14% first-name typos, 9% surname typos, 5% missing DOBs, 15% email variations, 11% city abbreviations, 7% gender errors.

---

## PDF report sections

1. Dataset information and completeness chart
2. Blocking rules and cumulative comparison count chart
3. Comparison methods
4. Model training with match weights chart and parameter estimates chart
5. Unlinkable records chart
6. Edge metrics and match weight histogram
7. Cluster metrics and dataset overlap Venn diagram
8. Confusion matrix with Precision-Recall curve and CRL score

---

## Known limitations and planned work

- Composite blocking rules currently limited to pairs of fields
- SAIL Databank provisioning on the export page is a placeholder for full deployment
- The EDA correlation check uses value-level co-occurrence for text fields, not statistical correlation; this is intentional and appropriate for record linkage use cases
- Upload mode currently accepts CSV only; DuckDB, Parquet, and Excel support is planned

---

## Related repositories

- linkage-workflow: JSON-driven Splink model configuration and notebook templates
- linkage-metrics: DuckDB SQL metric functions for intra- and inter-model linkage quality assessment
