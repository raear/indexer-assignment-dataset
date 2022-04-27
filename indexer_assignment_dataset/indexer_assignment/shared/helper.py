from ...shared.helper import to_date


def enumerate_year_week_numbers(start_date, end_date):
    start_year = start_date.year
    start_week = start_date.isocalendar()[1]
    end_year = end_date.year
    end_week = end_date.isocalendar()[1]

    for year in range(start_year, end_year + 1):
        range_start = 1
        range_end = 54
        
        if year == start_year:
            range_start = start_week
        if year == end_year:
            range_end = end_week + 1

        for week in range(range_start, range_end):
            yield year, week

    
def filter_dataset_by_weeknumber(dataset, year, week_number):
    # filtered_dataset = [example for example in dataset if to_date(example["date_completed"]).isocalendar()[1] == week_number ]
    filtered_dataset = []
    for example in dataset:
        date_completed_date = to_date(example["date_completed"])
        date_completed_year = date_completed_date.year
        date_completed_week_number = date_completed_date.isocalendar()[1]
        if (date_completed_year == year 
            and date_completed_week_number == week_number):
            filtered_dataset.append(example)
    return filtered_dataset