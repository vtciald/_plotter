import pandas as pd
import numpy as np
from ezdata.processor import DataProcessor
from ezdata import prep
import re

def test_clean_arg():
    
    dp = DataProcessor()

    test_string = 'Hello\xa0World–'
    test_list = [['“Derp’', 'This\u00A0is\u202fbad'], 'Hello\xa0World–']
    test_dict = {'“Derp’': 25, 'Hello\xa0World–': 'Bad\u2014String\u00A0Value'}

    str_result = dp.clean_arg(test_string)
    list_result = dp.clean_arg(test_list)
    dict_result = dp.clean_arg(test_dict)
    
    assert str_result == 'Hello World-'
    assert list_result == [['"Derp\'', 'This is bad'], 'Hello World-']
    assert dict_result == {'"Derp\'': 25, 'Hello World-': 'Bad-String Value'}

def test_clean_df():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        'Col2': [25, 50]
    })

    result = dp.clean_df(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_clean_df_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    result = dp.clean_df(test_df, cols = ['“Derp’', 'Col2'])

    pd.testing.assert_frame_equal(result, expected)

def test_clean_df_cols_regex():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        '"Test\'': ['Bad-String Value', 'Hello World-'],
        'Col2': [25, 50]
    })

    result = dp.clean_df(test_df, pattern = r'e')

    pd.testing.assert_frame_equal(result, expected)

def test_remove_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [10, 20],
        'Col2': [25, 50],
        'test_Other_0': ['a string would', 'likely appear here']
    })

    expected = pd.DataFrame({
        'Col1': [10, 20],
        'Col2': [25, 50]
    })

    result = dp.remove_cols(test_df, pattern = dp.PATTERN_ALIDA_OTHER_OE)

    pd.testing.assert_frame_equal(result, expected)

def test_remove_verbal_anchors():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': [1, 2, 3, 4, 5],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64',
        'Col3': 'Int64'
    })

    result = dp.remove_verbal_anchors(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_remove_verbal_anchors_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64'
    })

    result = dp.remove_verbal_anchors(test_df, cols = ['Col1', 'Col2'])

    pd.testing.assert_frame_equal(result, expected)

def test_recode_values():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 'Three'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    mapper = {
        'Not at.*': 1,
        'Somewhat': 2,
        'Totally': 'Three',
    }
    result = dp.recode_vals(test_df, mapper, regex_keys = True, suffix = '1')

    pd.testing.assert_frame_equal(result, expected)

def test_recode_values_new_col():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    expected = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
        'rc_Col1': [1, 2, 'Three'],
    })

    mapper = {
        'Not at.*': 1,
        'Somewhat': 2,
        'Totally': 'Three',
    }
    result = dp.recode_vals(test_df, mapper, new_col_prefix = 'rc_', regex_keys = True, suffix = '1')

    pd.testing.assert_frame_equal(result, expected)

def test_recode_values_extract():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64'
    })

    result = dp.remove_verbal_anchors(test_df, cols = ['Col1', 'Col2'])

    pd.testing.assert_frame_equal(result, expected)

def test_filter_straightliners():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, 3, 4, 5],
        'Col2': [np.nan, np.nan, 3, 7, 7],
        'Col3': [np.nan, np.nan, 6, 6, 6]
    })

    result = dp.filter_straightliners(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_filter_straightliners_min_unique():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, np.nan, 4, 5],
        'Col2': [np.nan, np.nan, np.nan, 7, 7],
        'Col3': [np.nan, np.nan, np.nan, 6, 6]
    })

    result = dp.filter_straightliners(test_df, min_unique = 3)

    pd.testing.assert_frame_equal(result, expected)

def test_filter_straightliners_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6],
        'Col4': [1, 1, 1, 1, 1]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, 3, 4, 5],
        'Col2': [np.nan, np.nan, 3, 7, 7],
        'Col3': [np.nan, np.nan, 6, 6, 6],
        'Col4': [1, 1, 1, 1, 1]
    })

    result = dp.filter_straightliners(test_df, cols = ['Col1', 'Col2', 'Col3'])

    pd.testing.assert_frame_equal(result, expected)

def test_bin_i():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': ['(0.992, 3.0]', '(0.992, 3.0]', '(0.992, 3.0]', '(5.0, 7.0]', '(5.0, 7.0]', '(5.0, 7.0]', '(7.0, 9.0]', '(7.0, 9.0]'],
        'Col2': ['(0.949, 13.75]', '(13.75, 26.5]', '(13.75, 26.5]', '(26.5, 39.25]', '(39.25, 52.0]', '(39.25, 52.0]', '(39.25, 52.0]', '(39.25, 52.0]']
    })

    result = dp.bin(test_df, 'i4')

    pd.testing.assert_frame_equal(result, expected)

def test_bin_q():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': ['(0.999, 3.0]', '(0.999, 3.0]', '(0.999, 3.0]', '(3.0, 6.0]', '(3.0, 6.0]', '(3.0, 6.0]', '(6.75, 9.0]', '(6.75, 9.0]'],
        'Col2': ['(0.999, 22.75]', '(0.999, 22.75]', '(22.75, 37.0]', '(22.75, 37.0]', '(37.0, 50.25]', '(37.0, 50.25]', '(50.25, 52.0]', '(50.25, 52.0]']
    })

    result = dp.bin(test_df, 'q4')

    pd.testing.assert_frame_equal(result, expected)

def test_bin_list():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': ['(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]'],
        'Col2': ['(0, 30]', '(0, 30]', '(0, 30]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]']
    })

    result = dp.bin(test_df, [0, 30, 60])
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_bounds():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, np.nan, 6, 6, 6, 9, 9],
        'Col2': [np.nan, 16, 25, 33, 41, 50, np.nan, np.nan]
    })

    result = dp.filter_by_bounds(test_df, min_val = 5, max_val = 50)
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_iqr():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, np.nan]
    })

    result = dp.filter_by_iqr(test_df)
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_iqr_factor():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    result = dp.filter_by_iqr(test_df, factor = 1.6)
    
    pd.testing.assert_frame_equal(result, expected)

def test_get_cols_prefix():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one', 'Test_col2']
    _, result_cols = prep._prep_args(test_df, prefix = 'Test')

    assert expected_cols == result_cols

def test_get_cols_suffix():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Col4_test', 'Col5_test']
    _, result_cols = prep._prep_args(test_df, suffix = 'test')

    assert expected_cols == result_cols

def test_get_cols_pattern_regex():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col2', 'test_col3', 'Col4_test', 'Col5_test', 'Col6']

    pattern = re.compile(r'col\d', re.IGNORECASE)

    _, result_cols = prep._prep_args(test_df, pattern = pattern)

    assert expected_cols == result_cols

def test_get_cols_multi():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6_nope': [16, 17, 18],
        'Some_Col7_test': [16, 17, 18],
        'Col8_testing': [16, 17, 18],
    })

    expected_cols = ['Col4_test', 'Col5_test']

    pattern = re.compile(r'\d_')

    _, result_cols = prep._prep_args(test_df, prefix = 'Col', pattern = pattern, suffix = 'test')

    assert expected_cols == result_cols

def test_rename_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        '4_t': 'New label 2'
    }

    expected = pd.DataFrame({
        'New label 1': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    result = dp.rename_cols(test_df, mapper)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_regex_arg():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        '4_t': 'New label 2'
    }

    expected = pd.DataFrame({
        'New label 1': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'New label 2': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    result = dp.rename_cols(test_df, mapper, regex_keys = True)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_regex_pattern():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        re.compile('4_t'): 'New label 2'
    }

    expected = pd.DataFrame({
        'New label 1': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'New label 2': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    result = dp.rename_cols(test_df, mapper)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_duplicate():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        re.compile(r'col\d', re.IGNORECASE): 'New label 2'
    }

    data_rows = [
        [1, 4, 7, 10, 13, 16],
        [2, 5, 8, 11, 14, 17],
        [3, 6, 9, 12, 15, 18],
    ]
    columns = [
        'New label 1', 'New label 2', 'New label 2', 'New label 2', 'New label 2', 'New label 2',
    ]

    expected = pd.DataFrame(
        data_rows,
        columns = columns
    )

    result = dp.rename_cols(test_df, mapper)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_extract():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1: Extra stuff here': ['Not at all', 'Somewhat', 'Totally'],
        'Col2: Something else': ['Not at all', 'Somewhat', 'Totally'],
    })

    expected = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    mapper_regex = r'(^.+):.+'
    result = dp.rename_cols(test_df, mapper_regex)

    pd.testing.assert_frame_equal(result, expected)

# Standardizing characters in arguments
test_clean_arg()

# Standardizing characters in dfs
test_clean_df()
test_clean_df_cols()
test_clean_df_cols_regex()

# Removing columns from dfs
test_remove_cols()

# Removing verbal anchors from values
test_remove_verbal_anchors()
test_remove_verbal_anchors_cols()

# Filtering straightliners
test_filter_straightliners()
test_filter_straightliners_min_unique()
test_filter_straightliners_cols()

# Binning values
test_bin_i()
test_bin_q()
test_bin_list()

# Filtering by bounds
test_filter_by_bounds()

# Filtering by IQR
test_filter_by_iqr()
test_filter_by_iqr_factor()

# Test getting column names
test_get_cols_prefix()
test_get_cols_suffix()
test_get_cols_pattern_regex()
test_get_cols_multi()

# Test renaming columns
test_rename_cols()
test_rename_cols_regex_arg()
test_rename_cols_regex_pattern()
test_rename_cols_duplicate()
test_rename_cols_extract()

# Test recoding values
test_recode_values()
test_recode_values_new_col()