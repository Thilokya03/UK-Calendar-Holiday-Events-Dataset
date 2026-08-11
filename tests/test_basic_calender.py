import os
from src.calendar.basic_calender import DATA_DIR, PROCESSED_DATA_DIR


def test_directory_structure():
    # Check if the data directory exists
    assert os.path.exists(DATA_DIR), \
        f"Data directory {DATA_DIR} does not exist."

    # Check if the processed data directory exists
    assert os.path.exists(PROCESSED_DATA_DIR), \
        f"Processed data directory {PROCESSED_DATA_DIR} does not exist."

    # Check if the basic_calendar.csv file exists
    basic_calendar_file = PROCESSED_DATA_DIR / "basic_calendar.csv"

    assert os.path.exists(basic_calendar_file), \
        f"Basic calendar file {basic_calendar_file} does not exist."
        
        

def test_basic_calendar_file_content():
    basic_calendar_file = PROCESSED_DATA_DIR / "basic_calendar.csv"
    
    # Check if the basic_calendar.csv file exists
    assert os.path.exists(basic_calendar_file), \
        f"Basic calendar file {basic_calendar_file} does not exist."
    
    assert basic_calendar_file.stat().st_size > 0, \
        f"Basic calendar file {basic_calendar_file} is empty."
    
    assert basic_calendar_file.suffix == ".csv", \
        f"Basic calendar file {basic_calendar_file} is not a CSV file."
    
    assert basic_calendar_file.name == "basic_calendar.csv", \
        f"Basic calendar file {basic_calendar_file} does not have the expected name 'basic_calendar.csv'."
        
    assert basic_calendar_file.parent == PROCESSED_DATA_DIR, \
        f"Basic calendar file {basic_calendar_file} is not located in the expected directory {PROCESSED_DATA_DIR}."
        
    assert basic_calendar_file.is_file(), \
        f"Basic calendar file {basic_calendar_file} is not a regular file."
    
    assert basic_calendar_file.exists(), \
        f"Basic calendar file {basic_calendar_file} does not exist."
    
    assert basic_calendar_file.is_absolute(), \
        f"Basic calendar file {basic_calendar_file} is not an absolute path."
    
    assert basic_calendar_file.is_relative_to(PROCESSED_DATA_DIR), \
        f"Basic calendar file {basic_calendar_file} is not relative to the expected directory {PROCESSED_DATA_DIR}."
        
        
