import pandas as pd
import numpy as np
import pytest
from plotter.processor import DataProcessor
import re

def test_fix_characters_arg():
    
    dp = DataProcessor()

    test_string = 'Hello\xa0World–'
    test_list = [['“Derp’', 'This\u00A0is\u202fbad'], 'Hello\xa0World–']
    test_dict = {'“Derp’': 25, 'Hello\xa0World–': 'Bad\u2014String\u00A0Value'}

    str_result = dp.fix_characters_arg(test_string)
    list_result = dp.fix_characters_arg(test_list)
    dict_result = dp.fix_characters_arg(test_dict)
    
    assert str_result == 'Hello World-'
    assert list_result == [['"Derp\'', 'This is bad'], 'Hello World-']
    assert dict_result == {'"Derp\'': 25, 'Hello World-': 'Bad-String Value'}

def test_fix_characters_df():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        'Col2': [25, 50]
    })

    result = dp.fix_characters_df(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_fix_characters_df_cols():

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

    result = dp.fix_characters_df(test_df, cols = ['“Derp’', 'Col2'])

    pd.testing.assert_frame_equal(result, expected)

def test_fix_characters_df_cols_regex():

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

    result = dp.fix_characters_df(test_df, pattern = r'e')

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

def test_validate_one_arg_used_error():

    dp = DataProcessor()

    with pytest.raises(ValueError) as info: 
        dp._validate_one_arg_used(one = None, two = None)

    assert 'One of the following parameters must be used' in str(info.value)

def test_validate_one_arg_used_warn():

    dp = DataProcessor()

    with pytest.warns(UserWarning) as info: 
        dp._validate_one_arg_used(one = 1, two = 2, three = None)

    assert len(info) == 1
    assert 'Only one of the following parameters can be used' in str(info[0].message)

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
    result_cols = dp._get_cols(test_df, prefix = 'Test')

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
    result_cols = dp._get_cols(test_df, suffix = 'test')

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

    result_cols = dp._get_cols(test_df, pattern = pattern)

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

    result_cols = dp._get_cols(test_df, prefix = 'Col', pattern = pattern, suffix = 'test')

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

def test_agg_cols_mean():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'NewCol4': [3, 1, 1, 2/3, 2/3, 1/3, 2/3, 1/3],
    })

    result = dp.agg_cols(test_df, 'mean', 'NewCol4')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_std():

    dp = DataProcessor()

    sqrt2 = np.sqrt(2)

    test_df = pd.DataFrame({
        'Col1': [0 * sqrt2, 1 * sqrt2,  2 * sqrt2,  3 * sqrt2, 0 * sqrt2,  4 * sqrt2,  5 * sqrt2, 10 * sqrt2],
        'Col2': [1 * sqrt2, 3 * sqrt2,  5 * sqrt2,  7 * sqrt2, 4 * sqrt2,  9 * sqrt2, 11 * sqrt2, 20 * sqrt2],
    })

    expected = pd.DataFrame({
        'Col1': [0 * sqrt2, 1 * sqrt2,  2 * sqrt2,  3 * sqrt2, 0 * sqrt2,  4 * sqrt2,  5 * sqrt2, 10 * sqrt2],
        'Col2': [1 * sqrt2, 3 * sqrt2,  5 * sqrt2,  7 * sqrt2, 4 * sqrt2,  9 * sqrt2, 11 * sqrt2, 20 * sqrt2],
        'NewCol3': [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 10.0],
    })

    result = dp.agg_cols(test_df, 'std', 'NewCol3')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_var():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 1, 10,  5,  0, 20,  4],
        'Col2': [3, 4, 6, 12, 11,  1, 10, 14],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 1, 10,  5,  0, 20,  4],
        'Col2': [3, 4, 6, 12, 11,  1, 10, 14],
        'NewCol3': [2.0, 2.0, 12.5, 2.0, 18.0, 0.5, 50.0, 50.0],
    })

    result = dp.agg_cols(test_df, 'var', 'NewCol3')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_prod():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'NewCol4': [7, 0, 1, 0, 0, 0, 0, 0],
    })

    result = dp.agg_cols(test_df, 'prod', 'NewCol4')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_nunique():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'NewCol4': [2, 3, 1, 2, 2, 2, 2, 2],
    })

    result = dp.agg_cols(test_df, 'nunique', 'NewCol4')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_count():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [np.nan, np.nan, 1, 1, 0, 0, 0, 0],
        'Col2': [np.nan, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, 1, 1, 0, 0, 0, 0],
        'Col2': [np.nan, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'NewCol4': [1, 2, 3, 3, 3, 3, 3, 3],
    })

    result = dp.agg_cols(test_df, 'count', 'NewCol4')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_sum():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [np.nan, np.nan, 1, 1, 0, 0, 0, 0],
        'Col2': [np.nan, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, 1, 1, 0, 0, 0, 0],
        'Col2': [np.nan, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'NewCol4': [7.0, 2.0, 3.0, 2.0, 2.0, 1.0, 2.0, 1.0],
    })

    result = dp.agg_cols(test_df, 'sum', 'NewCol4')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_median():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
        'NewCol5': [2.0, 1.5, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5],
    })

    result = dp.agg_cols(test_df, 'median', 'NewCol5')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_min():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
        'NewCol5': [1, 0, 1, 0, 0, 0, 0, 0],
    })

    result = dp.agg_cols(test_df, 'min', 'NewCol5')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_max():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
        'NewCol5': [7, 3, 1, 1, 1, 1, 1, 1],
    })

    result = dp.agg_cols(test_df, 'max', 'NewCol5')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_and():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 1],
        'NewCol5': [1, 0, 1, 0, 0, 0, 0, 0],
    })

    result = dp.agg_cols(test_df, 'and', 'NewCol5')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_or():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 0],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 0],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 0],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 0],
        'NewCol5': [1, 1, 1, 1, 1, 1, 1, 0],
    })

    result = dp.agg_cols(test_df, 'or', 'NewCol5')

    pd.testing.assert_frame_equal(result, expected)

def test_agg_cols_or_drop_inputs():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 0],
        'Col4': [3, 3, 1, 1, 1, 1, 1, 0],
    })

    expected = pd.DataFrame({
        'NewCol5': [1, 1, 1, 1, 1, 1, 1, 0],
    })

    result = dp.agg_cols(test_df, 'or', 'NewCol5', drop_inputs = True)

    pd.testing.assert_frame_equal(result, expected)

def test_agg_rows_list():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [0.5, 0.534522, 8.0],
        'Col2': [0.5, 0.534522, 8.0],
        'Col3': [1.875, 2.10017, 8.0],
    }, index = ['mean', 'std', 'count'])

    result = dp.agg_rows(test_df, ['mean', 'std', 'count'])

    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(result, expected)

def test_agg_rows_str():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [1, 0, 1, 0, 1, 0, 1, 0],
        'Col3': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.Series(
        [0.5, 0.5, 1.875],
        index = ['Col1', 'Col2', 'Col3']
    )

    result = dp.agg_rows(test_df, 'mean')

    assert isinstance(result, pd.Series)

    pd.testing.assert_series_equal(result, expected)

def test_compute_ci_z():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [0.5, 0.5345, 8, 0.1296, 0.8704],
        'Col2': [1.875, 2.1002, 8, 0.4197, 3.3303],
        },
        index = ['mean', 'std', 'count', 'lower', 'upper']
    )

    result = dp.compute_ci(test_df, how = 'z')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_wald():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame({
        'Col1': [0.5, np.nan, 8, 0.1535, 0.8465],
        },
        index = ['mean', 'std', 'count', 'lower', 'upper']
    )

    result = dp.compute_ci(test_df, how = 'wald', cols = 'Col1')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)


# Standardizing characters in arguments
test_fix_characters_arg()

# Standardizing characters in dfs
test_fix_characters_df()
test_fix_characters_df_cols()
test_fix_characters_df_cols_regex()

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

# Test one-arg validation
test_validate_one_arg_used_error()
test_validate_one_arg_used_warn()

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

# Test column aggregation
test_agg_cols_mean()
test_agg_cols_std()
test_agg_cols_var()
test_agg_cols_prod()
test_agg_cols_nunique()
test_agg_cols_count()
test_agg_cols_sum()
test_agg_cols_median()
test_agg_cols_min()
test_agg_cols_max()
test_agg_cols_and()
test_agg_cols_or()
test_agg_cols_or_drop_inputs()

# Test row aggregation
test_agg_rows_list()
test_agg_rows_str()

# Test computing CIs
test_compute_ci_z()
test_compute_ci_wald()