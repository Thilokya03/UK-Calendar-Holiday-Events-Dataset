import requests
import pandas as pd
import csv
import io
import re
from pathlib import Path


# ============================================================
# 1. SETTINGS
# ============================================================

START_YEAR = 2000

OUTPUT_DIR = Path("data/economic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. OFFICIAL ONS DATA SOURCES
# ============================================================

SOURCES = {

    # UK Industrial Production Index
    "industrial_production_index": {
        "series_id": "K222",
        "url": (
            "https://www.ons.gov.uk/generator?"
            "format=csv&uri=%2Feconomy%2Feconomicoutputandproductivity"
            "%2Foutput%2Ftimeseries%2Fk222%2Fdiop"
        )
    },

    # UK Monthly GDP / GVA Index
    "gdp_index": {
        "series_id": "ECY2",
        "url": (
            "https://www.ons.gov.uk/generator?"
            "format=csv&uri=%2Feconomy%2Fgrossdomesticproductgdp"
            "%2Ftimeseries%2Fecy2%2Fmgdp"
        )
    },

    # UK Consumer Price Index
    "cpi_index": {
        "series_id": "D7BT",
        "url": (
            "https://www.ons.gov.uk/generator?"
            "format=csv&uri=%2Feconomy%2Finflationandpriceindices"
            "%2Ftimeseries%2Fd7bt%2Fmm23"
        )
    },

    # UK Unemployment Rate
    "unemployment_rate": {
        "series_id": "MGSX",
        "url": (
            "https://www.ons.gov.uk/generator?"
            "format=csv&uri=%2Femploymentandlabourmarket"
            "%2Fpeoplenotinwork%2Funemployment"
            "%2Ftimeseries%2Fmgsx%2Flms"
        )
    }
}


# ============================================================
# 3. MONTH MAPPING
# ============================================================

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12
}


# ============================================================
# 4. FUNCTION TO DOWNLOAD AND EXTRACT MONTHLY ONS DATA
# ============================================================

def extract_ons_monthly_data(
    url,
    column_name,
    start_year=2010
):

    print("\n------------------------------------------")
    print(f"Downloading: {column_name}")
    print("------------------------------------------")

    # Send request to ONS
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=60
    )

    # Stop if download failed
    response.raise_for_status()

    print("Download successful.")

    # Read CSV text
    csv_content = io.StringIO(response.text)

    reader = csv.reader(csv_content)

    records = []

    # ONS file contains:
    #
    # metadata
    # annual data
    # quarterly data
    # monthly data
    #
    # Monthly rows look like:
    #
    # 2010 JAN,99.0
    # 2010 FEB,98.5
    # 2010 MAR,99.2

    monthly_pattern = re.compile(
        r"^(\d{4})\s([A-Z]{3})$"
    )

    for row in reader:

        if len(row) < 2:
            continue

        period = row[0].strip()
        value = row[1].strip()

        match = monthly_pattern.match(period)

        if not match:
            continue

        year = int(match.group(1))
        month_name = match.group(2)

        if year < start_year:
            continue

        if month_name not in MONTH_MAP:
            continue

        month_number = MONTH_MAP[month_name]

        # Convert value to numeric
        numeric_value = pd.to_numeric(
            value,
            errors="coerce"
        )

        # Create proper monthly date
        date = pd.Timestamp(
            year=year,
            month=month_number,
            day=1
        )

        records.append({
            "date": date,
            column_name: numeric_value
        })


    # Convert records to DataFrame
    df = pd.DataFrame(records)


    # Check if data was found
    if df.empty:

        raise ValueError(
            f"No monthly data found for {column_name}"
        )


    # Sort by date
    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )


    print(
        f"Extracted {len(df)} monthly records."
    )

    print(
        f"First date: {df['date'].min().date()}"
    )

    print(
        f"Last date: {df['date'].max().date()}"
    )

    return df


# ============================================================
# 5. EXTRACT INDUSTRIAL PRODUCTION
# ============================================================

industrial_df = extract_ons_monthly_data(
    url=SOURCES[
        "industrial_production_index"
    ]["url"],

    column_name="industrial_production_index",

    start_year=START_YEAR
)


# ============================================================
# 6. EXTRACT MONTHLY GDP
# ============================================================

gdp_df = extract_ons_monthly_data(
    url=SOURCES[
        "gdp_index"
    ]["url"],

    column_name="gdp_index",

    start_year=START_YEAR
)


# ============================================================
# 7. EXTRACT CPI
# ============================================================

cpi_df = extract_ons_monthly_data(
    url=SOURCES[
        "cpi_index"
    ]["url"],

    column_name="cpi_index",

    start_year=START_YEAR
)


# ============================================================
# 8. EXTRACT UNEMPLOYMENT RATE
# ============================================================

unemployment_df = extract_ons_monthly_data(
    url=SOURCES[
        "unemployment_rate"
    ]["url"],

    column_name="unemployment_rate",

    start_year=START_YEAR
)


# ============================================================
# 9. SAVE INDIVIDUAL DATASETS
# ============================================================

industrial_df.to_csv(
    OUTPUT_DIR / "industrial_production.csv",
    index=False
)

gdp_df.to_csv(
    OUTPUT_DIR / "monthly_gdp.csv",
    index=False
)

cpi_df.to_csv(
    OUTPUT_DIR / "cpi.csv",
    index=False
)

unemployment_df.to_csv(
    OUTPUT_DIR / "unemployment.csv",
    index=False
)


# ============================================================
# 10. MERGE ALL ECONOMIC DATA
# ============================================================

economic_df = industrial_df.merge(
    gdp_df,
    on="date",
    how="outer"
)

economic_df = economic_df.merge(
    cpi_df,
    on="date",
    how="outer"
)

economic_df = economic_df.merge(
    unemployment_df,
    on="date",
    how="outer"
)


# ============================================================
# 11. SORT DATA
# ============================================================

economic_df = (
    economic_df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 12. ADD DATE FEATURES
# ============================================================

economic_df["year"] = (
    economic_df["date"].dt.year
)

economic_df["month"] = (
    economic_df["date"].dt.month
)


# ============================================================
# 13. REARRANGE COLUMNS
# ============================================================

economic_df = economic_df[
    [
        "date",
        "year",
        "month",
        "industrial_production_index",
        "gdp_index",
        "cpi_index",
        "unemployment_rate"
    ]
]


# ============================================================
# 14. CHECK DUPLICATES
# ============================================================

duplicate_count = (
    economic_df["date"]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate dates:",
    duplicate_count
)


# ============================================================
# 15. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    economic_df
    .isna()
    .sum()
)


# ============================================================
# 16. CHECK FINAL DATA
# ============================================================

print("\nFirst 12 rows:")

print(
    economic_df.head(12)
)


print("\nLast 12 rows:")

print(
    economic_df.tail(12)
)


print("\nDataset shape:")

print(
    economic_df.shape
)


# ============================================================
# 17. SAVE FINAL DATASET
# ============================================================

FINAL_FILE = (
    OUTPUT_DIR /
    "uk_economic_data.csv"
)

economic_df.to_csv(
    FINAL_FILE,
    index=False
)


print(
    "\nFinal dataset saved to:"
)

print(
    FINAL_FILE.resolve()
)


print(
    "\nEconomic data extraction completed successfully."
)