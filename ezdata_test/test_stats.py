import pandas as pd
import numpy as np
from ezdata.processor import DataProcessor

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

    expected = pd.DataFrame(
        {
            'point_estimate': [0.5, 1.875],
            'lower': [0.1296, 0.4197],
            'upper': [0.8704, 3.3303],
            'count': [8, 8],
        },
        index = ['Col1', 'Col2'],
    )

    result = dp.calc_ci(test_df, method = 'z')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_t():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [0.5, 1.875],
            'lower': [0.0531, 0.1192],
            'upper': [0.9469, 3.6308],
            'count': [8, 8],
        },
        index = ['Col1', 'Col2'],
    )

    result = dp.calc_ci(test_df, method = 't')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_wald():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [0.5],
            'lower': [0.1535],
            'upper': [0.8465],
            'count': [8],
        },
        index = ['Col1'],
    )

    result = dp.calc_ci(test_df, method = 'wald', cols = 'Col1')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_wilson():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [7, 2, 1, 1, 1, 1, 1, 1],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [0.5],
            'lower': [0.2152],
            'upper': [0.7848],
            'count': [8],
        },
        index = ['Col1'],
    )

    result = dp.calc_ci(test_df, method = 'wilson', cols = 'Col1')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_ac():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [0.5333],
            'lower': [0.3011],
            'upper': [0.752],
            'count': [15],
        },
        index = ['Col1'],
    )

    result = dp.calc_ci(test_df, method = 'agresti_coull')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_cp():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [0.5333],
            'lower': [0.2659],
            'upper': [0.7873],
            'count': [15],
        },
        index = ['Col1'],
    )

    result = dp.calc_ci(test_df, method = 'clopper_pearson')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_jeffreys():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [0.5333],
            'lower': [0.2939],
            'upper': [0.7612],
            'count': [15],
        },
        index = ['Col1'],
    )

    result = dp.calc_ci(test_df, method = 'jeffreys')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_bootstrap_bca():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [5, 13, 73, 84, 65, 83, 85, 12, 32, 53, 85, 45, 47, 23, 57, 59, 60, 23],
        'Col2': [7, 3, 7, 9, 3, 1, 74, 62, 76, 6, 34, 68, 96, 34, 86, 90, 52, 745],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [50.222, 80.7222],
            'lower': [37.6752, 35.6667],
            'upper': [61.5, 207.4474],
            'count': [18, 18],
        },
        index = ['Col1', 'Col2'],
    )

    result = dp.calc_ci(test_df, method = 'bootstrap_bca')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_bootstrap_basic():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [5, 13, 73, 84, 65, 83, 85, 12, 32, 53, 85, 45, 47, 23, 57, 59, 60, 23],
        'Col2': [7, 3, 7, 9, 3, 1, 74, 62, 76, 6, 34, 68, 96, 34, 86, 90, 52, 745],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [50.222, 80.7222],
            'lower': [38.7208, -3.6736],
            'upper': [62.5569, 133.0014],
            'count': [18, 18],
        },
        index = ['Col1', 'Col2'],
    )

    result = dp.calc_ci(test_df, method = 'bootstrap_basic')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_compute_ci_bootstrap_percentile():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [5, 13, 73, 84, 65, 83, 85, 12, 32, 53, 85, 45, 47, 23, 57, 59, 60, 23],
        'Col2': [7, 3, 7, 9, 3, 1, 74, 62, 76, 6, 34, 68, 96, 34, 86, 90, 52, 745],
    })

    expected = pd.DataFrame(
        {
            'point_estimate': [50.222, 80.7222],
            'lower': [37.8875, 28.4431],
            'upper': [61.7236, 165.1181],
            'count': [18, 18],
        },
        index = ['Col1', 'Col2'],
    )

    result = dp.calc_ci(test_df, method = 'bootstrap_percentile')
    result = result.round(4)

    pd.testing.assert_frame_equal(result, expected)

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
test_compute_ci_t()
test_compute_ci_wald()
test_compute_ci_wilson()
test_compute_ci_ac()
test_compute_ci_cp()
test_compute_ci_jeffreys()
test_compute_ci_bootstrap_bca()
test_compute_ci_bootstrap_basic()
test_compute_ci_bootstrap_percentile()