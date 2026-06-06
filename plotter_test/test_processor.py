import pandas as pd
import numpy as np
import pytest
from plotter.processor import DataProcessor

def test_fix_characters_arg():
    
    dp = DataProcessor()

    bad_string = 'Hello\xa0World–'
    bad_list = [['“Derp’', 'This\u00A0is\u202fbad'], 'Hello\xa0World–']
    bad_dict = {'“Derp’': 25, 'Hello\xa0World–': 'Bad\u2014String\u00A0Value'}

    str_result = dp.fix_characters_arg(bad_string)
    list_result = dp.fix_characters_arg(bad_list)
    dict_result = dp.fix_characters_arg(bad_dict)
    
    assert str_result == 'Hello World-'
    assert list_result == [['"Derp\'', 'This is bad'], 'Hello World-']
    assert dict_result == {'"Derp\'': 25, 'Hello World-': 'Bad-String Value'}

def test_fix_characters_df():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected_df = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        'Col2': [25, 50]
    })

    result_df = dp.fix_characters_df(test_df)

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_fix_characters_df_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected_df = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    result_df = dp.fix_characters_df(test_df, cols = ['“Derp’', 'Col2'])

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_remove_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50],
        'test_Other_0': ['a string would', 'likely appear here']
    })

    expected_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    result_df = dp.remove_cols(test_df, dp.PATTERN_ALIDA_OTHER_OE)

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_remove_cols_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50],
        'test_Other_0': ['a string would', 'likely appear here'],
        'test2_Other_0': ['a string would', 'likely appear here']
    })

    expected_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50],
        'test2_Other_0': ['a string would', 'likely appear here']
    })

    result_df = dp.remove_cols(test_df, dp.PATTERN_ALIDA_OTHER_OE, cols = ['“Derp’', 'Col2', 'test_Other_0'])

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_remove_str_anchors():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': [1, 2, 3, 4, 5],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64',
        'Col3': 'Int64'
    })

    result_df = dp.remove_str_anchors(test_df)

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_remove_str_anchors_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64'
    })

    result_df = dp.remove_str_anchors(test_df, cols = ['Col1', 'Col2'])

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_straightliners():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, np.nan, 3, 4, 5],
        'Col2': [np.nan, np.nan, 3, 7, 7],
        'Col3': [np.nan, np.nan, 6, 6, 6]
    })

    result_df = dp.filter_straightliners(test_df)

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_straightliners_min_unique():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, np.nan, np.nan, 4, 5],
        'Col2': [np.nan, np.nan, np.nan, 7, 7],
        'Col3': [np.nan, np.nan, np.nan, 6, 6]
    })

    result_df = dp.filter_straightliners(test_df, min_unique = 3)

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_straightliners_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6],
        'Col4': [1, 1, 1, 1, 1]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, np.nan, 3, 4, 5],
        'Col2': [np.nan, np.nan, 3, 7, 7],
        'Col3': [np.nan, np.nan, 6, 6, 6],
        'Col4': [1, 1, 1, 1, 1]
    })

    result_df = dp.filter_straightliners(test_df, cols = ['Col1', 'Col2', 'Col3'])

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_bin_i():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected_df = pd.DataFrame({
        'Col1': ['(0.992, 3.0]', '(0.992, 3.0]', '(0.992, 3.0]', '(5.0, 7.0]', '(5.0, 7.0]', '(5.0, 7.0]', '(7.0, 9.0]', '(7.0, 9.0]'],
        'Col2': ['(0.949, 13.75]', '(13.75, 26.5]', '(13.75, 26.5]', '(26.5, 39.25]', '(39.25, 52.0]', '(39.25, 52.0]', '(39.25, 52.0]', '(39.25, 52.0]']
    })

    result_df = dp.bin(test_df, 'i4')

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_bin_q():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected_df = pd.DataFrame({
        'Col1': ['(0.999, 3.0]', '(0.999, 3.0]', '(0.999, 3.0]', '(3.0, 6.0]', '(3.0, 6.0]', '(3.0, 6.0]', '(6.75, 9.0]', '(6.75, 9.0]'],
        'Col2': ['(0.999, 22.75]', '(0.999, 22.75]', '(22.75, 37.0]', '(22.75, 37.0]', '(37.0, 50.25]', '(37.0, 50.25]', '(50.25, 52.0]', '(50.25, 52.0]']
    })

    result_df = dp.bin(test_df, 'q4')

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_bin_list():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected_df = pd.DataFrame({
        'Col1': ['(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]'],
        'Col2': ['(0, 30]', '(0, 30]', '(0, 30]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]']
    })

    result_df = dp.bin(test_df, [0, 30, 60])
    
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_bin_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected_df = pd.DataFrame({
        'Col1': ['(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]'],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    result_df = dp.bin(test_df, [0, 30, 60], cols = ['Col1'])
    
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_by_bounds():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, np.nan, np.nan, 6, 6, 6, 9, 9],
        'Col2': [np.nan, 16, 25, 33, 41, 50, np.nan, np.nan]
    })

    result_df = dp.filter_by_bounds(test_df, min_val = 5, max_val = 50)
    
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_by_bounds_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, np.nan, np.nan, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    result_df = dp.filter_by_bounds(test_df, min_val = 5, max_val = 50, cols = ['Col1'])
    
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_by_iqr():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, np.nan]
    })

    result_df = dp.filter_by_iqr(test_df)
    
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_by_iqr_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    result_df = dp.filter_by_iqr(test_df, cols = ['Col1'])
    
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_by_iqr_factor():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected_df = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    result_df = dp.filter_by_iqr(test_df, factor = 1.6)
    
    pd.testing.assert_frame_equal(result_df, expected_df)

# Standardizing characters in arguments
test_fix_characters_arg()

# Standardizing characters in dfs
test_fix_characters_df()
test_fix_characters_df_cols()

# Removing columns from dfs
test_remove_cols()
test_remove_cols_cols()

# Removing verbal anchors from values and coercing to Int64
test_remove_str_anchors()
test_remove_str_anchors_cols()

# Replacing straightliners' values with NaN
test_filter_straightliners()
test_filter_straightliners_min_unique()
test_filter_straightliners_cols()

# Binning values
test_bin_i()
test_bin_q()
test_bin_list()
test_bin_cols()

# Filtering by bounds
test_filter_by_bounds()
test_filter_by_bounds_cols()

# Filtering by IQR
test_filter_by_iqr()
test_filter_by_iqr_cols()
test_filter_by_iqr_factor()