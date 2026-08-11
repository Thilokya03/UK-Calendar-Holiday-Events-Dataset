import pandas as pd
import requests
import json
import logging
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

GOVUK_HOLIDAY_URL = "https://www.gov.uk/bank-holidays.json"


def check_directory_exists(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory {directory} created.")
        return

    print(f"Directory {directory} already exists.")


def fetch_govuk_holidays() -> dict:
    try:
        response = requests.get(
            GOVUK_HOLIDAY_URL,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching holiday data: {e}")
        return {}


def save_raw_holidays(holidays_data: dict):
    if not holidays_data:
        print("No holiday data fetched. Raw file not created.")
        return False

    raw_file_path = RAW_DATA_DIR / "bank_holidays.json"

    with open(raw_file_path, "w", encoding="utf-8") as f:
        json.dump(
            holidays_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Raw holiday data saved to {raw_file_path}.")

    return True


def parse_region_holidays(
    file_path: Path,
    region: str
) -> pd.DataFrame:

    with open(file_path, "r", encoding="utf-8") as f:
        holidays_data = json.load(f)

    if region not in holidays_data:
        logging.error(
            f"Region '{region}' not found in holiday data."
        )
        return pd.DataFrame()

    events = holidays_data[region]["events"]

    holidays_list = []

    for event in events:
        holidays_list.append({
            "date": event.get("date"),
            "holiday_name": event.get("title"),
            "notes": event.get("notes"),
            "bunting": event.get("bunting"),
            "region": region
        })

    holidays_df = pd.DataFrame(holidays_list)

    holidays_df["date"] = pd.to_datetime(
        holidays_df["date"]
    )

    return holidays_df


def combine_holiday_dataframes(
    dfs: list[pd.DataFrame]
) -> pd.DataFrame:

    combined_df = pd.concat(
        dfs,
        ignore_index=True
    )

    combined_df.drop_duplicates(
        subset=[
            "date",
            "holiday_name",
            "region"
        ],
        inplace=True
    )

    combined_df.sort_values(
        by=["date", "region"],
        inplace=True
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "combined_holidays.csv"
    )

    combined_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Processed holiday data saved to {output_path}."
    )

    return combined_df


def main():

    check_directory_exists(DATA_DIR)
    check_directory_exists(RAW_DATA_DIR)
    check_directory_exists(PROCESSED_DATA_DIR)

    holidays_data = fetch_govuk_holidays()

    if not save_raw_holidays(holidays_data):
        print("Holiday pipeline stopped.")
        return

    raw_file_path = (
        RAW_DATA_DIR
        / "bank_holidays.json"
    )

    eng_wales_df = parse_region_holidays(
        raw_file_path,
        "england-and-wales"
    )

    sco_df = parse_region_holidays(
        raw_file_path,
        "scotland"
    )

    combined_df = combine_holiday_dataframes(
        [
            eng_wales_df,
            sco_df
        ]
    )

    print(combined_df.head())


if __name__ == "__main__":
    main()