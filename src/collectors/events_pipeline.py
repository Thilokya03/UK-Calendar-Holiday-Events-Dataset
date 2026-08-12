from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Change this to 2010-01-01 if you only need data
# matching the NESO demand period.
START_DATE = pd.Timestamp("2000-01-01")

END_DATE = pd.Timestamp.today().normalize()


# ------------------------------------------------------------
# Project root
#
# events_pipeline.py
# └── collectors
#     └── src
#         └── PROJECT ROOT
#
# Therefore parents[2] is correct.
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

RAW_EVENT_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "events"
)

SOURCE_DIR = (
    RAW_EVENT_DIR
    / "source"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


RAW_EVENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. RELIABLE DATA SOURCES
# ============================================================

# Official House of Commons Library
PARLIAMENT_ELECTION_URL = (
    "https://commonslibrary.parliament.uk/"
    "research-briefings/sn04512/"
)


# Oxford COVID-19 Government Response Tracker
OXFORD_COVID_URL = (
    "https://raw.githubusercontent.com/"
    "OxCGRT/"
    "covid-policy-dataset/"
    "main/"
    "data/"
    "OxCGRT_compact_national_v1.csv"
)


# Official England Football
ENGLAND_FOOTBALL_URL = (
    "https://www.englandfootball.com/"
    "england/"
    "mens-senior-team/"
    "Legacy"
)


# ============================================================
# 3. STANDARD EVENT FORMAT
# ============================================================

EVENT_COLUMNS = [
    "event_id",
    "event_name",
    "event_type",
    "start_date",
    "end_date",
    "start_time",
    "end_time",
    "all_day",
    "scope",
    "competition",
    "venue",
    "event_level",
    "source_name",
    "source_url",
]


# ============================================================
# 4. HTTP SESSION
# ============================================================

def create_session() -> requests.Session:
    """
    Create HTTP session with retry support.

    If a website temporarily returns errors such as:
        429
        500
        502
        503
        504

    the request will automatically retry.
    """

    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET"
        ],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; "
                "SmartGridForecaster/1.0; "
                "University research project)"
            )
        }
    )

    return session


SESSION = create_session()


# ============================================================
# 5. GENERAL HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace and special characters.
    """

    text = str(text)

    text = text.replace(
        "\xa0",
        " ",
    )

    text = text.replace(
        "–",
        "-",
    )

    text = text.replace(
        "—",
        "-",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def slugify(text: str) -> str:
    """
    Convert a string into an ID-friendly form.
    """

    text = normalize_text(text)

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def download_html(
    url: str,
) -> str:
    """
    Download an HTML page.
    """

    response = SESSION.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    return response.text


def download_file(
    url: str,
    output_path: Path,
) -> Path:
    """
    Download a file and store the original copy.

    Useful for preserving the raw source data.
    """

    print(
        f"Downloading:\n{url}"
    )

    with SESSION.get(
        url,
        timeout=180,
        stream=True,
    ) as response:

        response.raise_for_status()

        with open(
            output_path,
            "wb",
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(
                        chunk
                    )

    return output_path


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save DataFrame as CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved {len(df):,} rows -> {path}"
    )


def generate_event_ids(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate event IDs from event type, date and name.
    """

    df = df.copy()

    df["event_id"] = df.apply(
        lambda row: slugify(
            f"{row['event_type']}_"
            f"{row['start_date']}_"
            f"{row['event_name']}"
        ),
        axis=1,
    )

    # Handle duplicate IDs
    duplicate_number = (
        df.groupby("event_id")
        .cumcount()
    )

    duplicate_mask = (
        duplicate_number > 0
    )

    df.loc[
        duplicate_mask,
        "event_id"
    ] = (
        df.loc[
            duplicate_mask,
            "event_id"
        ]
        + "_"
        + duplicate_number[
            duplicate_mask
        ].astype(str)
    )

    return df


def standardize_event_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure every event dataset has
    exactly the same columns.
    """

    df = df.copy()

    for column in EVENT_COLUMNS:

        if column not in df.columns:

            df[column] = ""

    # ---------------------------------------------
    # Dates
    # ---------------------------------------------

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce",
    )

    df["end_date"] = pd.to_datetime(
        df["end_date"],
        errors="coerce",
    )

    # If no end date, event ends on start date.
    df["end_date"] = (
        df["end_date"]
        .fillna(
            df["start_date"]
        )
    )

    # Remove invalid dates.
    df = df.dropna(
        subset=[
            "start_date"
        ]
    )

    # ---------------------------------------------
    # Date range filter
    # ---------------------------------------------

    df = df[
        (
            df["start_date"]
            >= START_DATE
        )
        &
        (
            df["start_date"]
            <= END_DATE
        )
    ]

    # ---------------------------------------------
    # Convert dates to YYYY-MM-DD
    # ---------------------------------------------

    df["start_date"] = (
        df["start_date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    df["end_date"] = (
        df["end_date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    df = generate_event_ids(
        df
    )

    return df[
        EVENT_COLUMNS
    ]


# ============================================================
# 6. ELECTION COLLECTOR
# ============================================================

def find_election_table(
    html: str,
) -> pd.DataFrame:
    """
    Find the historical UK General Election table
    on the House of Commons Library page.

    No election dates are manually entered here.
    """

    tables = pd.read_html(
        StringIO(html)
    )

    best_table = None
    best_score = 0

    for table in tables:

        if table.shape[1] < 2:
            continue

        first_column = (
            table.iloc[:, 0]
            .astype(str)
            .map(normalize_text)
        )

        score = (
            first_column
            .str.contains(
                r"\b(18|19|20)\d{2}\b",
                regex=True,
                na=False,
            )
            .sum()
        )

        if score > best_score:

            best_score = score
            best_table = table

    if (
        best_table is None
        or best_score < 10
    ):

        raise RuntimeError(
            "Could not identify the election table "
            "on the House of Commons website."
        )

    return best_table


def parse_election_date(
    year: int,
    date_text: str,
):
    """
    Parse an election date from the official table.

    Modern elections have a single polling day.

    Older elections may contain ranges, but our
    project begins much later than that period.
    """

    date_text = normalize_text(
        date_text
    )

    # Remove text in brackets.
    date_text = re.sub(
        r"\([^)]*\)",
        "",
        date_text,
    )

    # If a range exists, take first polling date.
    if " to " in date_text.lower():

        date_text = re.split(
            r"\s+to\s+",
            date_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

    # Remove weekday.
    date_text = re.sub(
        (
            r"^(Monday|Tuesday|Wednesday|"
            r"Thursday|Friday|Saturday|Sunday)"
            r"\s+"
        ),
        "",
        date_text,
        flags=re.IGNORECASE,
    )

    # Add year if not present.
    if not re.search(
        r"\b\d{4}\b",
        date_text,
    ):

        date_text = (
            f"{date_text} {year}"
        )

    return pd.to_datetime(
        date_text,
        dayfirst=True,
        errors="coerce",
    )


def extract_elections_from_table(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the actual election records.
    """

    records = []

    for _, row in table.iterrows():

        values = [
            normalize_text(value)
            for value in row.tolist()
        ]

        if len(values) < 2:
            continue

        year_match = re.search(
            r"\b((?:18|19|20)\d{2})\b",
            values[0],
        )

        if not year_match:
            continue

        year = int(
            year_match.group(1)
        )

        election_date = (
            parse_election_date(
                year,
                values[1],
            )
        )

        if pd.isna(
            election_date
        ):
            continue

        records.append(
            {
                "event_name":
                    f"UK General Election {year}",

                "event_type":
                    "general_election",

                "start_date":
                    election_date,

                "end_date":
                    election_date,

                "start_time":
                    "",

                "end_time":
                    "",

                "all_day":
                    1,

                "scope":
                    "United Kingdom",

                "competition":
                    "",

                "venue":
                    "United Kingdom",

                "event_level":
                    "national",

                "source_name":
                    "House of Commons Library",

                "source_url":
                    PARLIAMENT_ELECTION_URL,
            }
        )

    return pd.DataFrame(
        records
    )


def extract_general_elections():
    """
    Complete election extraction process.
    """

    print(
        "\n====================================="
    )

    print(
        "1. GENERAL ELECTIONS"
    )

    print(
        "====================================="
    )

    print(
        "Downloading House of Commons page..."
    )

    html = download_html(
        PARLIAMENT_ELECTION_URL
    )

    # Save original HTML
    html_path = (
        SOURCE_DIR
        / "parliament_elections.html"
    )

    html_path.write_text(
        html,
        encoding="utf-8",
    )

    print(
        f"Raw HTML saved -> {html_path}"
    )

    table = find_election_table(
        html
    )

    elections = (
        extract_elections_from_table(
            table
        )
    )

    elections = (
        standardize_event_dataframe(
            elections
        )
    )

    if elections.empty:

        raise RuntimeError(
            "Election extraction returned zero rows."
        )

    elections = (
        elections
        .sort_values(
            "start_date"
        )
        .reset_index(
            drop=True
        )
    )

    save_dataframe(
        elections,
        RAW_EVENT_DIR
        / "elections.csv",
    )

    print(
        "\nElection dates extracted:"
    )

    print(
        elections[
            [
                "start_date",
                "event_name",
            ]
        ].to_string(
            index=False
        )
    )

    return elections


# ============================================================
# 7. COVID POLICY COLLECTOR
# ============================================================

def find_column_by_prefix(
    df: pd.DataFrame,
    prefix: str,
) -> str:
    """
    Find a column even if Oxford later adds
    small suffix changes to the dataset.
    """

    matches = [
        column
        for column in df.columns
        if column.startswith(
            prefix
        )
    ]

    if not matches:

        raise RuntimeError(
            f"Column beginning with "
            f"'{prefix}' was not found."
        )

    return matches[0]


def load_oxford_data():
    """
    Download and load the original Oxford CSV.
    """

    print(
        "\n====================================="
    )

    print(
        "2. COVID POLICY DATA"
    )

    print(
        "====================================="
    )

    covid_raw_path = (
        SOURCE_DIR
        / "OxCGRT_compact_national_v1.csv"
    )

    download_file(
        OXFORD_COVID_URL,
        covid_raw_path,
    )

    print(
        "Reading Oxford dataset..."
    )

    df = pd.read_csv(
        covid_raw_path,
        low_memory=False,
    )

    print(
        f"Rows downloaded: {len(df):,}"
    )

    return df


def extract_uk_covid_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract United Kingdom records.
    """

    required_columns = [
        "CountryCode",
        "Date",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise RuntimeError(
                f"Required Oxford column "
                f"'{column}' is missing."
            )

    uk = df[
        df["CountryCode"]
        == "GBR"
    ].copy()

    if uk.empty:

        raise RuntimeError(
            "No GBR records found "
            "in Oxford dataset."
        )

    # Oxford format is YYYYMMDD.
    uk["Date"] = pd.to_datetime(
        uk["Date"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        ),
        format="%Y%m%d",
        errors="coerce",
    )

    uk = uk.dropna(
        subset=[
            "Date"
        ]
    )

    return uk


def derive_covid_lockdown_periods(
    uk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine lockdown periods automatically.

    OxCGRT:
        C6 = Stay-at-home requirement

        0 = none
        1 = recommendation
        2 = mandatory with exceptions
        3 = mandatory with minimal exceptions

    We define lockdown as C6 >= 2.
    """

    c6_column = (
        find_column_by_prefix(
            uk,
            "C6M_Stay at home requirements",
        )
    )

    print(
        f"Using Oxford column: {c6_column}"
    )

    data = uk[
        [
            "Date",
            c6_column,
        ]
    ].copy()

    data = (
        data.groupby(
            "Date",
            as_index=False,
        )[c6_column]
        .max()
    )

    data = data.sort_values(["Date"])

    data["stay_home_level"] = (
        pd.to_numeric(
            data[c6_column],
            errors="coerce",
        )
    )

    data["is_lockdown"] = (
        data["stay_home_level"]
        >= 2
    ).astype(int)

    # ---------------------------------------------
    # Detect a new period if:
    #
    # 1. restriction value changes
    # OR
    # 2. date sequence has a gap
    # ---------------------------------------------

    previous_lockdown = (
        data["is_lockdown"]
        .shift()
    )

    date_gap = (
        data["Date"]
        .diff()
        .dt.days
    )

    new_period = (
        data["is_lockdown"]
        .ne(
            previous_lockdown
        )
        |
        date_gap.ne(1)
    )

    data["period_id"] = (
        new_period
        .cumsum()
    )

    lockdown_days = data[
        data["is_lockdown"] == 1
    ].copy()

    if lockdown_days.empty:

        raise RuntimeError(
            "No mandatory lockdown periods "
            "were found in Oxford data."
        )

    periods = (
        lockdown_days
        .groupby(
            "period_id"
        )
        .agg(
            start_date=(
                "Date",
                "min",
            ),

            end_date=(
                "Date",
                "max",
            ),

            event_level=(
                "stay_home_level",
                "max",
            ),
        )
        .reset_index(
            drop=True
        )
    )

    periods["event_name"] = (
        "Mandatory COVID-19 "
        "Stay-at-Home Requirement"
    )

    periods["event_type"] = (
        "covid_lockdown"
    )

    periods["start_time"] = ""
    periods["end_time"] = ""

    periods["all_day"] = 1

    periods["scope"] = (
        "United Kingdom"
    )

    periods["competition"] = ""
    periods["venue"] = ""

    periods["source_name"] = (
        "Oxford COVID-19 "
        "Government Response Tracker"
    )

    periods["source_url"] = (
        OXFORD_COVID_URL
    )

    return periods


def extract_covid_events():
    """
    Complete Oxford COVID extraction.
    """

    df = load_oxford_data()

    uk = extract_uk_covid_data(
        df
    )

    print(
        f"United Kingdom daily rows: "
        f"{len(uk):,}"
    )

    lockdowns = (
        derive_covid_lockdown_periods(
            uk
        )
    )

    lockdowns = (
        standardize_event_dataframe(
            lockdowns
        )
    )

    lockdowns = (
        lockdowns
        .sort_values(
            "start_date"
        )
        .reset_index(
            drop=True
        )
    )

    save_dataframe(
        lockdowns,
        RAW_EVENT_DIR
        / "covid_restrictions.csv",
    )

    print(
        "\nLockdown periods discovered "
        "directly from Oxford data:"
    )

    print(
        lockdowns[
            [
                "start_date",
                "end_date",
                "event_level",
            ]
        ].to_string(
            index=False
        )
    )

    return lockdowns


# ============================================================
# 8. FOOTBALL COLLECTOR
# ============================================================

GAME_PATTERN = re.compile(
    r"^Game\s+(\d+)\s*:\s*(.+)$",
    re.IGNORECASE,
)


DATE_PATTERN = re.compile(
    (
        r"\b("
        r"\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"\s+\d{4}"
        r")\b"
    ),
    re.IGNORECASE,
)


def football_page_lines(
    html: str,
) -> list[str]:
    """
    Convert England Football HTML into
    normalized text lines.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    lines = []

    for value in soup.stripped_strings:

        text = normalize_text(
            value
        )

        if text:

            lines.append(
                text
            )

    return lines


def split_game_blocks(
    lines: list[str],
):
    """
    Split the official page into:

        Game 1001 ...
        competition
        date
        venue
        ...

    blocks.
    """

    blocks = []

    current_block = []

    for line in lines:

        if GAME_PATTERN.match(
            line
        ):

            if current_block:

                blocks.append(
                    current_block
                )

            current_block = [
                line
            ]

        elif current_block:

            current_block.append(
                line
            )

    if current_block:

        blocks.append(
            current_block
        )

    return blocks


def identify_major_competition(
    block: list[str],
):
    """
    Keep only major England tournament matches.

    Included:
        FIFA World Cup final tournament
        World Cup Finals
        UEFA EURO final tournament
        European Championship Finals

    Excluded:
        qualifiers
        friendlies
        Nations League
    """

    for line in block[1:]:

        lower = line.lower()

        # ---------------------------------------------
        # Remove qualifying matches
        # ---------------------------------------------

        if (
            "qualifier" in lower
            or "qualifying" in lower
        ):
            continue

        # ---------------------------------------------
        # FIFA World Cup
        # ---------------------------------------------

        if (
            "world cup finals"
            in lower
        ):

            return line

        if (
            "fifa world cup"
            in lower
            and "qualifier"
            not in lower
        ):

            return line

        # ---------------------------------------------
        # EURO
        # ---------------------------------------------

        if (
            "european championship finals"
            in lower
        ):

            return line

        if (
            "uefa euro"
            in lower
            and "qualifier"
            not in lower
        ):

            return line

    return None


def find_date_in_block(
    block: list[str],
):
    """
    Search the whole block for a date.

    We search all lines because some historical
    records contain dates inside explanatory text.
    """

    for line in block:

        match = DATE_PATTERN.search(
            line
        )

        if match:

            parsed = pd.to_datetime(
                match.group(1),
                dayfirst=True,
                errors="coerce",
            )

            if not pd.isna(
                parsed
            ):

                return parsed

    return None


def find_venue_in_block(
    block: list[str],
    match_date,
):
    """
    Try to extract the venue/location associated
    with the match date.

    Example source text:

        12 June 2010 - Rustenburg

    We do not use strftime here because formats
    such as %-d are not supported on Windows.
    """

    if match_date is None:
        return ""

    for line in block:

        date_match = DATE_PATTERN.search(
            line
        )

        if not date_match:
            continue

        parsed_date = pd.to_datetime(
            date_match.group(1),
            dayfirst=True,
            errors="coerce",
        )

        if pd.isna(parsed_date):
            continue

        # Make sure this date is actually
        # the match date we are looking for.
        if parsed_date.normalize() != match_date.normalize():
            continue

        # Everything following the date may
        # contain the venue.
        remainder = (
            line[
                date_match.end():
            ]
            .strip()
        )

        # Remove separators such as:
        #
        # 12 June 2010 - Rustenburg
        #                ^
        remainder = re.sub(
            r"^[\s\-–—,:]+",
            "",
            remainder,
        ).strip()

        # Avoid accidentally storing an entire
        # paragraph as the venue.
        if (
            remainder
            and len(remainder) <= 100
        ):
            return remainder

    return ""

def parse_football_blocks(
    blocks,
) -> pd.DataFrame:
    """
    Extract major tournament matches.
    """

    records = []

    for block in blocks:

        game_match = GAME_PATTERN.match(
            block[0]
        )

        if not game_match:
            continue

        game_number = int(
            game_match.group(1)
        )

        result_text = normalize_text(
            game_match.group(2)
        )

        competition = (
            identify_major_competition(
                block
            )
        )

        if competition is None:
            continue

        match_date = (
            find_date_in_block(
                block
            )
        )

        if match_date is None:
            continue

        if (
            match_date < START_DATE
            or match_date > END_DATE
        ):
            continue

        venue = (
            find_venue_in_block(
                block,
                match_date,
            )
        )

        records.append(
            {
                "event_name":
                    f"England Match - {result_text}",

                "event_type":
                    "major_football",

                "start_date":
                    match_date,

                "end_date":
                    match_date,

                # Historical archive does not
                # consistently supply kickoff time.
                "start_time":
                    "",

                "end_time":
                    "",

                "all_day":
                    1,

                "scope":
                    "Great Britain",

                "competition":
                    competition,

                "venue":
                    venue,

                "event_level":
                    "national",

                "source_name":
                    "England Football / "
                    "The Football Association",

                "source_url":
                    ENGLAND_FOOTBALL_URL,

                "_game_number":
                    game_number,
            }
        )

    return pd.DataFrame(
        records
    )


def extract_football_events():
    """
    Complete official football extraction.
    """

    print(
        "\n====================================="
    )

    print(
        "3. MAJOR FOOTBALL EVENTS"
    )

    print(
        "====================================="
    )

    print(
        "Downloading England Football archive..."
    )

    html = download_html(
        ENGLAND_FOOTBALL_URL
    )

    html_path = (
        SOURCE_DIR
        / "england_football_legacy.html"
    )

    html_path.write_text(
        html,
        encoding="utf-8",
    )

    print(
        f"Raw HTML saved -> {html_path}"
    )

    lines = football_page_lines(
        html
    )

    print(
        f"Page text lines: {len(lines):,}"
    )

    blocks = split_game_blocks(
        lines
    )

    print(
        f"Game blocks found: {len(blocks):,}"
    )

    matches = parse_football_blocks(
        blocks
    )

    if matches.empty:

        raise RuntimeError(
            "No major England football matches "
            "were extracted.\n"
            "The England Football website "
            "may have changed its structure."
        )

    if "_game_number" in matches.columns:

        matches = matches.sort_values(
            [
                "start_date",
                "_game_number",
            ]
        )

        matches = matches.drop(
            columns=[
                "_game_number"
            ]
        )

    matches = (
        standardize_event_dataframe(
            matches
        )
    )

    matches = (
        matches
        .drop_duplicates(
            subset=[
                "start_date",
                "event_name",
            ]
        )
        .sort_values(
            "start_date"
        )
        .reset_index(
            drop=True
        )
    )

    save_dataframe(
        matches,
        RAW_EVENT_DIR
        / "football.csv",
    )

    print(
        "\nMajor football matches extracted:"
    )

    print(
        matches[
            [
                "start_date",
                "event_name",
                "competition",
            ]
        ].to_string(
            index=False
        )
    )

    return matches


# ============================================================
# 9. COMBINE ALL EVENTS
# ============================================================

def combine_events(
    elections: pd.DataFrame,
    covid: pd.DataFrame,
    football: pd.DataFrame,
):
    """
    Combine all event sources.
    """

    print(
        "\n====================================="
    )

    print(
        "4. COMBINING EVENT DATA"
    )

    print(
        "====================================="
    )

    events = pd.concat(
        [
            elections,
            covid,
            football,
        ],
        ignore_index=True,
        sort=False,
    )

    events["start_date_sort"] = (
        pd.to_datetime(
            events["start_date"],
            errors="coerce",
        )
    )

    events = (
        events
        .drop_duplicates(
            subset=[
                "event_name",
                "event_type",
                "start_date",
                "end_date",
            ]
        )
        .sort_values(
            [
                "start_date_sort",
                "event_type",
            ]
        )
        .drop(
            columns=[
                "start_date_sort"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_path = (
        PROCESSED_DIR
        / "events.csv"
    )

    save_dataframe(
        events,
        output_path,
    )

    return events


# ============================================================
# 10. EXPAND EVENT DATE RANGES
# ============================================================

def expand_events_to_days(
    events: pd.DataFrame,
):
    """
    Example:

    lockdown:
        2020-03-23 -> 2020-05-10

    becomes:

        2020-03-23
        2020-03-24
        2020-03-25
        ...
        2020-05-10

    This is required before joining with
    daily/hourly electricity-demand data.
    """

    rows = []

    for _, event in events.iterrows():

        start_date = pd.to_datetime(
            event["start_date"],
            errors="coerce",
        )

        end_date = pd.to_datetime(
            event["end_date"],
            errors="coerce",
        )

        if pd.isna(
            start_date
        ):
            continue

        if pd.isna(
            end_date
        ):
            end_date = start_date

        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        for date in date_range:

            rows.append(
                {
                    "date":
                        date,

                    "event_id":
                        event[
                            "event_id"
                        ],

                    "event_name":
                        event[
                            "event_name"
                        ],

                    "event_type":
                        event[
                            "event_type"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. CREATE DAILY ML FEATURES
# ============================================================

def create_daily_event_features(
    events: pd.DataFrame,
):
    """
    Create one row for every calendar date.

    Output example:

    date
    is_event_day
    is_general_election
    is_covid_lockdown
    is_major_football
    event_count
    event_names
    """

    print(
        "\n====================================="
    )

    print(
        "5. CREATING DAILY FEATURES"
    )

    print(
        "====================================="
    )

    calendar = pd.DataFrame(
        {
            "date":
                pd.date_range(
                    start=START_DATE,
                    end=END_DATE,
                    freq="D",
                )
        }
    )

    long_events = (
        expand_events_to_days(
            events
        )
    )

    if long_events.empty:

        raise RuntimeError(
            "No event days available."
        )

    # ---------------------------------------------
    # Event count
    # ---------------------------------------------

    event_count = (
        long_events
        .groupby("date")
        .agg(
            event_count=(
                "event_id",
                "nunique",
            ),

            event_names=(
                "event_name",
                lambda values:
                    " | ".join(
                        sorted(
                            set(values)
                        )
                    ),
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------
    # One-hot event types
    # ---------------------------------------------

    flags = pd.crosstab(
        long_events["date"],
        long_events["event_type"],
    )

    flags = (
        flags > 0
    ).astype(int)

    flags.columns = [
        f"is_{slugify(column)}"
        for column in flags.columns
    ]

    flags = flags.reset_index()

    # ---------------------------------------------
    # Merge into full calendar
    # ---------------------------------------------

    features = calendar.merge(
        event_count,
        on="date",
        how="left",
    )

    features = features.merge(
        flags,
        on="date",
        how="left",
    )

    # ---------------------------------------------
    # Fill normal days
    # ---------------------------------------------

    features["event_count"] = (
        features["event_count"]
        .fillna(0)
        .astype(int)
    )

    features["event_names"] = (
        features["event_names"]
        .fillna("")
    )

    flag_columns = [
        column
        for column
        in features.columns
        if column.startswith(
            "is_"
        )
    ]

    for column in flag_columns:

        features[column] = (
            features[column]
            .fillna(0)
            .astype(int)
        )

    features["is_event_day"] = (
        features["event_count"]
        > 0
    ).astype(int)

    # ---------------------------------------------
    # Date formatting
    # ---------------------------------------------

    features["date"] = (
        features["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output_path = (
        PROCESSED_DIR
        / "event_daily_features.csv"
    )

    save_dataframe(
        features,
        output_path,
    )

    return features


# ============================================================
# 12. VALIDATION
# ============================================================

def validate_events(
    events: pd.DataFrame,
):
    """
    Basic data-quality checks.
    """

    print(
        "\n====================================="
    )

    print(
        "6. VALIDATION"
    )

    print(
        "====================================="
    )

    if events.empty:

        raise ValueError(
            "Event dataset is empty."
        )

    # ---------------------------------------------
    # Missing values
    # ---------------------------------------------

    if events[
        "start_date"
    ].isna().any():

        raise ValueError(
            "Missing event start dates found."
        )

    # ---------------------------------------------
    # Date order
    # ---------------------------------------------

    start_dates = pd.to_datetime(
        events["start_date"],
        errors="coerce",
    )

    end_dates = pd.to_datetime(
        events["end_date"],
        errors="coerce",
    )

    invalid_ranges = events[
        end_dates < start_dates
    ]

    if not invalid_ranges.empty:

        raise ValueError(
            "Invalid event date ranges found."
        )

    # ---------------------------------------------
    # Duplicate check
    # ---------------------------------------------

    duplicates = events.duplicated(
        subset=[
            "event_name",
            "event_type",
            "start_date",
        ]
    )

    if duplicates.any():

        print(
            "WARNING: Duplicate events exist."
        )

    print(
        f"Total events: {len(events):,}"
    )

    print(
        "\nEvents by type:"
    )

    print(
        events[
            "event_type"
        ].value_counts()
    )

    print(
        "\nDate range:"
    )

    print(
        events["start_date"].min(),
        "->",
        events["start_date"].max(),
    )

    print(
        "\nValidation successful."
    )


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main():

    print(
        "\n========================================="
    )

    print(
        "UK EVENT DATA COLLECTION PIPELINE"
    )

    print(
        "========================================="
    )

    print(
        f"Project root : {BASE_DIR}"
    )

    print(
        f"Start date   : {START_DATE.date()}"
    )

    print(
        f"End date     : {END_DATE.date()}"
    )

    print(
        f"Raw directory: {RAW_EVENT_DIR}"
    )

    # ========================================================
    # STEP 1
    # ========================================================

    elections = (
        extract_general_elections()
    )

    # ========================================================
    # STEP 2
    # ========================================================

    covid = (
        extract_covid_events()
    )

    # ========================================================
    # STEP 3
    # ========================================================

    football = (
        extract_football_events()
    )

    # ========================================================
    # STEP 4
    # ========================================================

    events = combine_events(
        elections,
        covid,
        football,
    )

    # ========================================================
    # STEP 5
    # ========================================================

    create_daily_event_features(
        events
    )

    # ========================================================
    # STEP 6
    # ========================================================

    validate_events(
        events
    )

    print(
        "\n========================================="
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "========================================="
    )

    print(
        "\nGenerated files:"
    )

    print(
        RAW_EVENT_DIR
        / "elections.csv"
    )

    print(
        RAW_EVENT_DIR
        / "covid_restrictions.csv"
    )

    print(
        RAW_EVENT_DIR
        / "football.csv"
    )

    print(
        PROCESSED_DIR
        / "events.csv"
    )

    print(
        PROCESSED_DIR
        / "event_daily_features.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()