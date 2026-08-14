# UK Calendar, Holiday, Events & Economic Dataset

A reproducible Python data pipeline for building **daily UK calendar, bank-holiday, event, and economic features** for time-series modelling and forecasting.

The project was created as part of the **Smart Grid Forecaster** data engineering workflow, where calendar and economic variables are used as external features for electricity-demand forecasting.

## What this repository provides

The pipeline collects, transforms, merges, validates, and releases two main analysis-ready datasets:

- `full_calendar_features.csv`
- `uk_economic_features_daily.csv`

Each release also includes:

- `SHA256SUMS.txt` for file-integrity verification

The final datasets are published through the repository's **GitHub Releases**.

## Main dataset outputs

### 1. `full_calendar_features.csv`

A continuous daily UK calendar dataset containing calendar, holiday, and event-related features.

Important fields include:

| Feature | Description |
|---|---|
| `date` | Calendar date |
| `year` | Year |
| `month_num` | Month number |
| `month_name` | Month name |
| `day` | Day of month |
| `day_of_week` | Day name |
| `day_of_week_num` | Numeric weekday |
| `week_of_year` | ISO week number |
| `quarter` | Calendar quarter |
| `is_weekend` | Weekend indicator |
| `season` | Winter, Spring, Summer, or Autumn |
| `holiday_names` | Holiday names for the date |
| `holiday_regions` | UK regions associated with the holiday |
| `is_bank_holiday_england_wales` | England and Wales bank-holiday flag |
| `is_bank_holiday_scotland` | Scotland bank-holiday flag |
| `is_bank_holiday` | Combined bank-holiday indicator |
| `is_non_working_day` | Weekend or bank-holiday indicator |
| `is_event_day` | Whether one or more tracked events occur |
| `event_count` | Number of tracked events on the date |
| `event_names` | Event names associated with the date |

The base calendar is generated from **2000 onward**, and the current implementation extends the calendar through **two years beyond the current year**.

### 2. `uk_economic_features_daily.csv`

A daily economic-feature dataset built from official UK monthly economic indicators.

The current pipeline includes:

| Feature | Source series |
|---|---|
| Industrial Production Index | ONS `K222` |
| Monthly GDP / GVA Index | ONS `ECY2` |
| Consumer Price Index | ONS `D7BT` |
| Unemployment Rate | ONS `MGSX` |

The daily dataset is designed for time-series modelling and includes lagged economic features so that a model does not use information that would not yet have been available at prediction time.

Key output fields include:

- `date`
- `year`
- `month`
- `year_month`
- `economic_reference_month`
- `industrial_production_index_lag1m`
- `gdp_index_lag1m`
- `cpi_index_lag1m`
- `unemployment_rate_lag1m`
- `economic_data_complete`

## Data sources

This project combines data from several external sources.

### Calendar and bank holidays

- **GOV.UK Bank Holidays API**  
  https://www.gov.uk/bank-holidays.json

- **UK Debt Management Office (DMO)** historical bank-holiday series  
  https://www.dmo.gov.uk/

### Economic indicators

Economic variables are collected from the **UK Office for National Statistics (ONS)**:

- Industrial Production Index
- Monthly GDP / GVA Index
- Consumer Price Index
- Unemployment Rate

https://www.ons.gov.uk/

### Event data

The event pipeline uses sources including:

- **House of Commons Library** for UK election information  
  https://commonslibrary.parliament.uk/

- **Oxford COVID-19 Government Response Tracker (OxCGRT)** for pandemic-policy events  
  https://github.com/OxCGRT/covid-policy-dataset

- **England Football** for selected national football events  
  https://www.englandfootball.com/

Source metadata is preserved where possible in the event-processing pipeline.

## Repository structure

```text
UK-Calendar-Holiday-Events-Dataset/
│
├── .github/
│   └── workflows/
│       ├── CI.yaml
│       └── CD.yaml
│
├── src/
│   ├── calendar/
│   │   └── basic_calender.py
│   │
│   ├── collectors/
│   │   ├── holiday_collector.py
│   │   ├── historical_holiday_collector.py
│   │   ├── events_pipeline.py
│   │   └── extract_economic_data.py
│   │
│   └── merge/
│       ├── merge_calendar_features.py
│       └── merge_economic_features.py
│
├── tests/
│   ├── test_basic_calender.py
│   └── test_merged_datasets.py
│
├── requirements.txt
├── LICENSE
├── DATA_LICENSE.md
└── README.md
```

Generated data is written under the `data/` directory during pipeline execution.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Thilokya03/UK-Calendar-Holiday-Events-Dataset.git
cd UK-Calendar-Holiday-Events-Dataset
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The CI pipeline currently uses **Python 3.11**.

## Running the pipeline locally

Run the pipeline components in the same order used by CI.

### Step 1 — Generate the base calendar

```bash
python src/calendar/basic_calender.py
```

### Step 2 — Collect bank-holiday data

```bash
python src/collectors/holiday_collector.py
python src/collectors/historical_holiday_collector.py
```

### Step 3 — Collect event data

```bash
python src/collectors/events_pipeline.py
```

### Step 4 — Collect economic data

```bash
python src/collectors/extract_economic_data.py
```

### Step 5 — Merge calendar features

```bash
python src/merge/merge_calendar_features.py
```

### Step 6 — Merge economic features

```bash
python src/merge/merge_economic_features.py
```

### Step 7 — Validate the generated datasets

```bash
python -m pytest tests/ -v
```

After a successful run, the two final datasets should exist at:

```text
data/processed/full_calendar_features.csv
data/processed/uk_economic_features_daily.csv
```

## Validation

The test suite checks important data-quality requirements, including:

- required columns
- valid dates
- one row per day
- no duplicate dates
- continuous daily coverage
- binary indicator consistency
- calendar-feature consistency
- expected economic-feature columns
- missing-value conditions

The release workflow only publishes datasets after validation succeeds.

## CI/CD

GitHub Actions is used to automate dataset generation and release management.

### Continuous Integration

The CI workflow:

1. checks out the repository
2. sets up Python 3.11
3. installs dependencies
4. generates the calendar
5. collects holiday data
6. collects event data
7. collects economic data
8. merges the datasets
9. runs the test suite
10. verifies the final CSV files
11. generates SHA-256 checksums
12. uploads the validated dataset artifact

The scheduled CI pipeline runs every **Saturday at 00:00 UTC**.

### Continuous Delivery

The CD workflow publishes validated dataset artifacts as versioned GitHub Releases.

Release assets include the final CSV datasets and SHA-256 checksums for integrity checking.

## Verify downloaded datasets

After downloading a release, compare the files against `SHA256SUMS.txt`.

Linux/macOS:

```bash
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell, a file can be checked with:

```powershell
Get-FileHash .\full_calendar_features.csv -Algorithm SHA256
```

Compare the result with the corresponding value in `SHA256SUMS.txt`.

## Versioning

Dataset releases use version tags such as:

```text
v1.0.0
v1.0.1
v1.1.0
```

For reproducible experiments, record the release tag used when training or evaluating a model.

## Intended use

The datasets are designed primarily for:

- electricity-demand forecasting
- energy analytics
- time-series machine learning
- calendar-effect analysis
- economic-feature engineering
- academic and research projects

They may also be useful for other UK forecasting tasks where holidays, events, and economic conditions can influence a target variable.

## Limitations

- External sources may change structure or availability.
- Historical and current source coverage can differ.
- Economic indicators are originally published at monthly frequency and are transformed for use in the daily feature dataset.
- Event coverage is selective rather than a complete record of every UK event.
- Future calendar rows can contain deterministic calendar features, while source-dependent holiday, event, and economic information depends on data availability.
- Users should independently verify source terms and suitability before using the datasets for legal, financial, operational, or other high-stakes decisions.

## Licensing

This repository uses separate licensing for software and data.

### Source code

The original source code in this repository is licensed under the **MIT License**.

See:

```text
LICENSE
```

### Dataset and database compilation

The project's original dataset compilation, schema, transformations, and derived features are made available under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, to the extent that the project contributors have the right to license them.

See:

```text
DATA_LICENSE.md
```

### Third-party source data

This repository incorporates or derives information from third-party sources. **The project does not relicense third-party material that it does not own.** Original source licences, copyright notices, attribution requirements, database rights, and terms of use continue to apply.

In particular, users should review the terms of the original providers before redistributing source-derived records.

## Attribution

If you reuse the project's original compiled dataset or derived features, a suitable attribution is:

> UK Calendar, Holiday, Events & Economic Dataset, Thilokya03 and contributors, GitHub repository, accessed from the relevant versioned release.

If your work uses source-derived information, also provide any attribution required by the original source provider.

## Contributing

Contributions that improve data quality, validation, documentation, source reliability, or feature engineering are welcome.

A typical contribution workflow is:

```text
feature branch
    ↓
pull request
    ↓
CI generation and validation
    ↓
review
    ↓
merge to main
```

Please avoid committing secrets, API keys, generated temporary files, or unverified datasets.

## Disclaimer

The datasets are provided for research, educational, and analytical use without a guarantee of completeness, accuracy, or fitness for a particular purpose.

Always verify important values against the authoritative source when the information is used for operational or high-stakes decisions.
