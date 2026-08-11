import pandas as pd
import requests
import logging
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DMO_HOLIDAY_URL = (
    "https://www.dmo.gov.uk/media/3lgd2zqc/"
    "ukbankholidays-nov23a.xls"
)

START_YEAR = 1998
END_YEAR = 2018


# --------------------------------------------------
# Directory setup
# --------------------------------------------------

def check_directory_exists(directory: Path):

    if not directory.exists():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Directory {directory} created."
        )

        return

    print(
        f"Directory {directory} already exists."
    )


# --------------------------------------------------
# Download historical Excel file
# --------------------------------------------------

def fetch_dmo_holidays() -> bool:

    try:

        response = requests.get(
            DMO_HOLIDAY_URL,
            timeout=30
        )

        response.raise_for_status()

        raw_file_path = (
            RAW_DATA_DIR
            / "dmo_bank_holidays.xls"
        )

        # Excel file is binary data
        with open(raw_file_path, "wb") as file:

            file.write(response.content)

        print(
            f"DMO holiday file saved to "
            f"{raw_file_path}"
        )

        return True

    except requests.exceptions.RequestException as error:

        logging.error(
            f"Error downloading DMO holidays: {error}"
        )

        return False


# --------------------------------------------------
# Read and extract holiday dates
# --------------------------------------------------

def extract_dmo_holidays(
    file_path: Path,
    start_year: int,
    end_year: int
) -> pd.DataFrame:

    # The DMO file does not have a normal
    # dataframe-style header.
    raw_df = pd.read_excel(
        file_path,
        header=None,
        engine="xlrd"
    )

    print("\nRaw DMO data:")
    print(raw_df.head(10))

    print(
        "\nRaw shape:",
        raw_df.shape
    )

    # Holiday dates are stored in the first column.
    date_column = raw_df.iloc[:, 0]

    # Try converting values to dates.
    dates = pd.to_datetime(
        date_column,
        errors="coerce"
    )

    # Keep only valid dates.
    holiday_df = pd.DataFrame({
        "date": dates
    })

    holiday_df.dropna(
        subset=["date"],
        inplace=True
    )

    # Filter required historical period.
    holiday_df = holiday_df[
        (
            holiday_df["date"].dt.year
            >= start_year
        )
        &
        (
            holiday_df["date"].dt.year
            <= end_year
        )
    ].copy()

    # Add metadata.
    holiday_df["source"] = (
        "UK Debt Management Office"
    )

    holiday_df["source_type"] = (
        "historical_bank_holiday_series"
    )

    # Sort by date.
    holiday_df.sort_values(
        by="date",
        inplace=True
    )

    # Remove exact duplicated dates.
    holiday_df.drop_duplicates(
        subset=["date"],
        inplace=True
    )

    holiday_df.reset_index(
        drop=True,
        inplace=True
    )

    return holiday_df


# --------------------------------------------------
# Validate extracted data
# --------------------------------------------------

def validate_historical_holidays(
    holiday_df: pd.DataFrame
) -> bool:

    if holiday_df.empty:

        logging.error(
            "Historical holiday dataset is empty."
        )

        return False

    if holiday_df["date"].isna().any():

        logging.error(
            "Missing dates found."
        )

        return False

    if holiday_df["date"].duplicated().any():

        logging.error(
            "Duplicate dates found."
        )

        return False

    print(
        f"\nNumber of historical holidays: "
        f"{len(holiday_df)}"
    )

    print(
        "First holiday:",
        holiday_df["date"].min()
    )

    print(
        "Last holiday:",
        holiday_df["date"].max()
    )

    print(
        "\nHistorical holiday validation passed."
    )

    return True


# --------------------------------------------------
# Save processed historical data
# --------------------------------------------------

def save_processed_holidays(
    holiday_df: pd.DataFrame
):

    output_path = (
        PROCESSED_DATA_DIR
        / "dmo_historical_holidays.csv"
    )

    holiday_df.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d"
    )

    print(
        f"\nHistorical holidays saved to "
        f"{output_path}"
    )


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    check_directory_exists(
        DATA_DIR
    )

    check_directory_exists(
        RAW_DATA_DIR
    )

    check_directory_exists(
        PROCESSED_DATA_DIR
    )

    # ---------------------------------------------
    # Step 1 - Download
    # ---------------------------------------------

    success = fetch_dmo_holidays()

    if not success:

        print(
            "Historical holiday pipeline stopped."
        )

        return

    raw_file_path = (
        RAW_DATA_DIR
        / "dmo_bank_holidays.xls"
    )

    # ---------------------------------------------
    # Step 2 - Extract
    # ---------------------------------------------

    historical_df = extract_dmo_holidays(
        raw_file_path,
        START_YEAR,
        END_YEAR
    )

    # ---------------------------------------------
    # Step 3 - Validate
    # ---------------------------------------------

    if not validate_historical_holidays(
        historical_df
    ):

        print(
            "Historical holiday validation failed."
        )

        return

    # ---------------------------------------------
    # Step 4 - Save
    # ---------------------------------------------

    save_processed_holidays(
        historical_df
    )

    print("\nExtracted holidays:")

    print(
        historical_df.head(20)
    )


if __name__ == "__main__":
    main()