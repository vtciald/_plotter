import pandas as pd
import numpy as np
from ezdata.processor import DataProcessor
from ezdata import test

def test_one_sample_t():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [-1, 0, 1, 0, 1, 0, -2, 0],
        'Col3': [10, -1, 2, 3, 4, 1, 2, 1]
    })

    expected = pd.DataFrame(
        {
            'mean_diff': [0.5, -0.125, 2.75],
            'p_value': [0.0331, 0.7318, 0.0497],
            'stat_sig': [True, False, True],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 't')

    result['mean_diff'] = result['mean_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_one_sample_wilcoxon():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [-1, 0, 1, 0, 1, 0, -2, 0],
        'Col3': [10, -1, 2, 3, 4, 1, 2, 1]
    })

    expected = pd.DataFrame(
        {
            'median_diff': [0.5, 0.0, 2.0],
            'p_value': [0.125, 1.0, 0.0312],
            'stat_sig': [False, False, True],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 'wilcoxon')

    result['median_diff'] = result['median_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_one_sample_sign():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [-1, 0, 1, 0, 1, 0, -2, 0],
        'Col3': [10, -1, 2, 3, 4, 1, 2, 1]
    })

    expected = pd.DataFrame(
        {
            'median_diff': [0.5, 0.0, 2.0],
            'p_value': [0.125, 1.0, 0.0703],
            'stat_sig': [False, False, False],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 'sign')

    result['median_diff'] = result['median_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_one_sample_sign_proportion():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'Col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'Col3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    })

    expected = pd.DataFrame(
        {
            'prop_diff': [0.0, 0.25, -0.4167],
            'p_value': [1.0, 0.146, 0.0063],
            'stat_sig': [False, False, True],
            'count': [12, 12, 12],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample_proportion(test_df, 'sign')

    result['prop_diff'] = result['prop_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

# Test one sample methods
test_one_sample_t()
test_one_sample_wilcoxon()
test_one_sample_sign()
test_one_sample_sign_proportion()