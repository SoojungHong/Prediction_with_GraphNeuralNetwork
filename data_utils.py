# libraries
import pandas as pd
from typing import Any, Union, Dict
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
import collections


# functions
def convert_to_date(unix_time):
    result_ms = pd.to_datetime(unix_time, unit='ms')
    str(result_ms)
    return result_ms


def is_columns_same(df, column1, column2):
    is_same = df[column1].equals(df[column2])
    return is_same


def count_string_frequency(input_all_strings):
    str_count: Dict[Any, int] = dict()
    for str in input_all_strings:
        if str in str_count:
            str_count[str] += 1
        else:
            str_count[str] = 1

    for key, value in str_count.items():
        print("% s : % d" % (key, value))

    return str_count


def n_most_common_in_series(series, n):
    d = collections.Counter(series)
    n_most_common = d.most_common(n)
    return n_most_common


def check_contain_nan(df):
    df.isnull().any()
    df.isnull().sum().sum() # This returns an integer of the total nummber of NaN values
    nan_rows = df[df['name column'].isnull()]   # To find out which rows have NaNs in a specific
    nan_rows = df[df.isnull().any(1)]   # To find out which rows have NaNs


def remove_special_char(df):
    df = df.apply(lambda row: row.replace('|', ''))
    df = df.apply(lambda row: row.replace('[', ''))
    df = df.apply(lambda row: row.replace(']', ''))
    df = df.apply(lambda row: row.replace('"', ''))
    df = df.apply(lambda row: row.replace('*', ''))
    return df


def remove_special_char_and_blank(df):
    df = df.apply(lambda row: row.replace('|', ''))
    df = df.apply(lambda row: row.replace('[', ''))
    df = df.apply(lambda row: row.replace(']', ''))
    df = df.apply(lambda row: row.replace('"', ''))
    df = df.apply(lambda row: row.replace('*', ''))
    df = df.apply(lambda row: row.strip())
    return df



def find_level_1_category_code(category_code):
    index = category_code.find('-')
    level_1_category_code = category_code[0:index]
    return int(level_1_category_code)


def merge_data_and_category(data_df, category_df, level_1_category_df):
    merged_data_df = pd.merge(left=data_df, right=category_df, on='category_id')
    print('test ')
    print(merged_data_df)

    merged_labels = ['place_name', 'category_name', 'category_id']
    merged_data_df[merged_labels]

    merged_data_df['Level_1_Code'] = merged_data_df.apply(lambda row: find_level_1_category_code(row.category_id), axis=1)
    final_merged_df = pd.merge(left=merged_data_df, right=level_1_category_df, on='Level_1_Code')
    return final_merged_df
