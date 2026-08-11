import pandas as pd
import datetime as dt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "processed"

END_YEAR =  dt.date.today().year +2
START_YEAR = 2000

SEASON_MAPPING = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
    12: "Winter"
}


def check_directory_exists(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory {directory} created.")\
    
    print(f"Directory {directory} already exists.")


def create_basic_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    date_range = pd.date_range(start=start_date, end=end_date)
    calendar_df = pd.DataFrame(date_range, columns=["date"])
    
    calendar_df["year"] = calendar_df["date"].dt.year
    calendar_df["month_num"] = calendar_df["date"].dt.month
    calendar_df["month_name"] = calendar_df["date"].dt.month_name()
    calendar_df["day"] = calendar_df["date"].dt.day
    calendar_df["day_of_week"] = calendar_df["date"].dt.day_name()
    calendar_df["day_of_week_num"] = calendar_df["date"].dt.dayofweek
    calendar_df["week_of_year"] = calendar_df["date"].dt.isocalendar().week
    calendar_df["quarter"] = calendar_df["date"].dt.quarter
    calendar_df["is_weekend"] = calendar_df["date"].dt.dayofweek >= 5
    calendar_df["season"] = calendar_df["date"].dt.month.map(SEASON_MAPPING)

    print(f"Basic calendar created from {start_date} to {end_date}.")
    return calendar_df

def main():
    check_directory_exists(DATA_DIR)
    check_directory_exists(RAW_DATA_DIR)
    
    create_basic_calendar(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31").to_csv(RAW_DATA_DIR / "basic_calendar.csv", index=False)
    
if __name__ == "__main__":
    main()