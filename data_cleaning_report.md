# Data Cleaning Report

## Overview

This report documents the cleaning pipeline applied to `resume_data.csv` (9,544 resume-job pairs, 34 columns) and the resulting `resume_data_cleaned.csv` (9,544 rows, 17 columns). The goal is to prepare data for predicting `matched_score` (0-1) using NLP models that compare resume and job description text.

---

## 1. Raw Data Summary

- **Rows**: 9,544
- **Columns**: 34 (33 after dropping duplicate `responsibilities.1`)
- **Target**: `matched_score` (float, 0-1 range, 0 nulls)
- **Column name issues fixed**:
  - `educationaL_requirements` -> `educational_requirements`
  - `experiencere_requirement` -> `experience_requirement`
  - `related_skils_in_job` -> `related_skills_in_job`
- All string values lowercased and stripped of leading/trailing whitespace
- BOM character removed from column headers

---

## 2. Placeholder-to-Null Normalization

The raw data uses many placeholder values instead of proper nulls. These were all converted to `NaN`:

### Global Placeholders Removed
- Empty strings, `n/a`, `none`, `[none]`, `[]`
- Lists containing only placeholders: `['n/a', 'n/a']`, `[none, none, none]`, etc.

### Column-Specific Placeholders
| Column | Placeholder | Rows Nullified |
|--------|-------------|----------------|
| `professional_company_names` | `['company name', 'company name', ...]` | 3,882 |
| `locations` | `['city, state', 'city, state', ...]` | 2,106 |

### Impact on Missingness (Before -> After Normalization)

Several columns had their true missingness revealed:

| Column | Before (%) | After (%) | Notes |
|--------|-----------|-----------|-------|
| `extra_curricular_organization_links` | 64.1 | 100.0 | Entirely placeholders |
| `company_urls` | 0.9 | 99.7 | Nearly all `[none]` |
| `expiry_dates` | 79.0 | 98.5 | |
| `issue_dates` | 79.0 | 93.6 | |
| `certification_skills` | 79.0 | 91.2 | |
| `professional_company_names` | 0.9 | 42.4 | Most were `'company name'` filler |
| `locations` | 0.9 | 69.1 | Most were `'city, state'` filler |
| `related_skills_in_job` | 0.9 | 13.5 | |
| `skills` | 0.6 | 1.2 | Slight increase from empty list cleanup |

---

## 3. Column Selection

### Dropped (14 columns, >80% missing or structurally useless)

| Column | Missing % | Reason |
|--------|-----------|--------|
| `languages` | 92.7 | Too sparse |
| `proficiency_levels` | 93.8 | Too sparse |
| `address` | 91.8 | Too sparse |
| `expiry_dates` | 98.5 | Too sparse |
| `issue_dates` | 93.6 | Too sparse |
| `online_links` | 100.0 | All null |
| `certification_skills` | 91.2 | Too sparse |
| `certification_providers` | 86.0 | Too sparse |
| `company_urls` | 99.7 | All null |
| `educational_results` | 79.1 | Borderline, mostly placeholders |
| `result_types` | 83.5 | Too sparse |
| `extra_curricular_organization_links` | 100.0 | All null |
| `start_dates` | 1.8 | Not useful for matching |
| `end_dates` | 3.5 | Not useful for matching |

### Kept (20 columns)

| Column | Null % | Unique | Side |
|--------|--------|--------|------|
| `career_objective` | 50.3 | 171 | resume |
| `skills` | 1.2 | 339 | resume |
| `educational_institution_name` | 1.2 | 324 | resume |
| `degree_names` | 1.2 | 176 | resume |
| `passing_years` | 12.9 | 141 | resume |
| `major_field_of_studies` | 10.9 | 202 | resume |
| `professional_company_names` | 42.4 | 188 | resume |
| `positions` | 1.5 | 298 | resume |
| `locations` | 69.1 | 67 | resume |
| `responsibilities` | 0.0 | 28 | resume |
| `related_skills_in_job` | 13.5 | 289 | resume |
| `extra_curricular_activity_types` | 66.7 | 83 | resume |
| `extra_curricular_organization_names` | 74.4 | 84 | resume |
| `role_positions` | 72.9 | 85 | resume |
| `job_position_name` | 0.0 | 28 | job |
| `educational_requirements` | 0.0 | 20 | job |
| `experience_requirement` | 14.3 | 17 | job |
| `age_requirement` | 42.8 | 14 | job |
| `skills_required` | 17.8 | 23 | job |
| `matched_score` | 0.0 | 345 | target |

---

## 4. List-as-String Parsing

12 columns stored Python lists as literal strings (e.g., `"['python', 'mysql']"`). A safe parser using `ast.literal_eval` with a regex fallback converted these to actual Python lists, filtering out remaining placeholder elements (`n/a`, `none`, `company name`, `city, state`).

### Parsed Columns and Sample Values

| Column | Sample Value |
|--------|-------------|
| `skills` | `['big data', 'hadoop', 'hive', 'python', ...]` |
| `educational_institution_name` | `['the amity school of engineering & technology (aset), noida']` |
| `degree_names` | `['b.tech']` |
| `passing_years` | `['2019']` |
| `major_field_of_studies` | `['electronics']` |
| `professional_company_names` | `['coca-cola']` |
| `related_skills_in_job` | `["['big data']"]` (nested string; see note below) |
| `positions` | `['big data analyst']` |
| `locations` | `['san jose, ca', 'milpitas, ca']` |
| `extra_curricular_activity_types` | `['professional organization', 'honor society', ...]` |
| `extra_curricular_organization_names` | `['ohio society of cpas', 'beta alpha psi', ...]` |
| `role_positions` | `['silver medal for economics junior award', ...]` |

**Note on `related_skills_in_job`**: This column has a nesting issue where some entries contain stringified lists within lists (e.g., `"['big data']"` as a list element). The values are still usable as text but aren't cleanly structured.

### Null % After Parsing

| Column | Null % |
|--------|--------|
| `extra_curricular_organization_names` | 74.4 |
| `role_positions` | 72.9 |
| `locations` | 70.5 |
| `extra_curricular_activity_types` | 66.7 |
| `career_objective` | 50.3 |
| `professional_company_names` | 43.9 |
| `age_requirement` | 42.8 |
| `skills_required` | 17.8 |
| `experience_requirement` | 14.3 |
| `related_skills_in_job` | 13.5 |
| `passing_years` | 12.9 |
| `major_field_of_studies` | 10.9 |
| `positions` | 1.5 |
| `degree_names` | 1.2 |
| `skills` | 1.2 |
| `educational_institution_name` | 1.2 |
| `responsibilities` | 0.0 |
| `job_position_name` | 0.0 |
| `educational_requirements` | 0.0 |
| `matched_score` | 0.0 |

---

## 5. Engineered Features

### Count Features

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `num_skills` | 21.6 | 19.3 | 0 | 144 |
| `num_degrees` | 1.6 | 1.0 | 0 | 11 |
| `num_positions` | 2.7 | 2.0 | 0 | 10 |
| `num_institutions` | 1.6 | 1.0 | 0 | 11 |

### Extracted Numeric Features

| Feature | Non-null Count | Source | Extraction |
|---------|---------------|--------|------------|
| `experience_years` | 8,180 / 9,544 | `experience_requirement` | Regex `(\d+)\s*year` |
| `age_min` | 3,411 / 9,544 | `age_requirement` | Regex `age\s*(\d+)\s*to\s*(\d+)` |
| `age_max` | 3,411 / 9,544 | `age_requirement` | Regex `age\s*(\d+)\s*to\s*(\d+)` |

### Skill Overlap

| Feature | Mean | Std | Max |
|---------|------|-----|-----|
| `skill_jaccard` | 0.003 | 0.015 | 0.273 |

The Jaccard similarity between resume `skills` (multi-word phrases like "machine learning") and `skills_required` (tokenized by whitespace into individual words) is very low. This is expected since exact string matching across different granularities fails. This motivates using embedding-based similarity in the modeling phase.

### Text Columns

| Column | Null Count | Composition |
|--------|-----------|-------------|
| `resume_text` | 0 | career_objective + skills + degree_names + major_field_of_studies + positions + responsibilities |
| `job_text` | 0 | job_position_name + educational_requirements + experience_requirement + skills_required |
| `combined_text` | 0 | resume_text + job_text |

**Sample `resume_text`**:
> big data analytics working and database warehouse manager with robust experience in handling all kinds of data. i have also used multiple cloud infrastructure services and am well acquainted with them...

**Sample `job_text`**:
> senior software engineer b.sc in computer science & engineering from a reputed university. at least 1 year...

---

## 6. Target Variable: `matched_score`

- **Range**: 0 to 1
- **Mean**: ~0.68
- **Nulls**: 0
- **Top values**: 0.85 (1,470 occurrences), 0.65 (1,321), 0.717 (516)
- The distribution is left-skewed, concentrated between 0.5 and 0.9

### Feature Correlations with `matched_score`

| Feature | Correlation |
|---------|------------|
| `age_max` | +0.256 |
| `num_positions` | +0.125 |
| `skill_jaccard` | +0.120 |
| `num_skills` | +0.102 |
| `num_degrees` | +0.064 |
| `num_institutions` | +0.044 |
| `age_min` | +0.007 |
| `experience_years` | -0.059 |

**Key observations**:
- `age_max` has the strongest correlation (+0.256), suggesting jobs with higher age ranges tend to have higher match scores
- `num_positions` (+0.125) and `skill_jaccard` (+0.120) show modest positive correlation
- `experience_years` is slightly negative (-0.059), which is counterintuitive and may reflect that more experienced roles are harder to match
- All correlations are weak, confirming that simple numeric features alone cannot predict match score well and that NLP-based text features are needed

---

## 7. Final Output: `resume_data_cleaned.csv`

### Shape: 9,544 rows x 17 columns

| Column | Type | Nulls | Purpose |
|--------|------|-------|---------|
| `matched_score` | float | 0 | Target variable |
| `resume_text` | str | 0 | NLP input (resume side) |
| `job_text` | str | 0 | NLP input (job side) |
| `combined_text` | str | 0 | NLP input (combined, for simple models) |
| `num_skills` | int | 0 | Structured feature |
| `num_degrees` | int | 0 | Structured feature |
| `num_positions` | int | 0 | Structured feature |
| `experience_years` | float | 1,364 | Structured feature |
| `age_min` | float | 6,133 | Structured feature |
| `age_max` | float | 6,133 | Structured feature |
| `skill_jaccard` | float | 0 | Structured feature |
| `job_position_name` | str | 0 | Raw job field (for grouping/filtering) |
| `educational_requirements` | str | 0 | Raw job field |
| `skills_required` | str | 1,701 | Raw job field |
| `skills` | str (JSON) | 112 | Parsed resume skills |
| `degree_names` | str (JSON) | 112 | Parsed degrees |
| `positions` | str (JSON) | 140 | Parsed positions |

### Sanity Checks Passed
- `matched_score`: 0 nulls
- `combined_text`: 0 nulls
- `resume_text`: 0 nulls
- `job_text`: 0 nulls
- All rows preserved (no rows dropped during cleaning)
- Spot-check confirmed parsed lists and text columns are coherent

---

## 8. Compatibility with Existing Baseline

The baseline model in `trying_regression.ipynb` uses `skills`, `job_position_name`, `educational_requirements`, and `responsibilities` via TF-IDF + LinearRegression (R²=0.477). The cleaned dataset supports this workflow:
- `combined_text` contains all of those fields and can be used directly
- `resume_text` and `job_text` enable comparison-based models (siamese networks, cross-encoders) that the baseline could not support
- Structured features (`num_skills`, `skill_jaccard`, `experience_years`, etc.) can supplement text features in hybrid models
