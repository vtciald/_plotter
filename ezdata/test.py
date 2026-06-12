import numpy as np
import pandas as pd
import scipy.stats
import re
from . import prep

def test_one_sample(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.0,
    alpha: float = 0.05,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'wilcoxon', 'sign', 'bootstrap'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the columns specified in the column-selection parameters.
            Columns include:
            - A descriptive difference column, dynamically named based on the test.
                * 'mean_diff' when `method = 't'`.
                * 'median_diff' when `method = 'wilcoxon' or `method = 'sign' and proportion = False`
                * 'prop_diff' when `method = 'sign' and proportion = True`
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    df, cols = prep._prep_args(df, cols, prefix, suffix, pattern)

    if method == 't':
        result = _one_sample_t(df, cols, null, alpha)

    elif method == 'wilcoxon':
        result = _one_sample_wilcoxon(df, cols, null, alpha)
    
    elif method == 'sign':
        result = _one_sample_sign(df, cols, null, alpha)
    
    elif method == 'bootstrap':
        raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result

def test_one_sample_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.5,
    alpha: float = 0.05,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'sign', 'bootstrap'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.5.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - 'prop_diff': The absolute difference between observed and null proportions.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.    """

    df, cols = prep._prep_args(df, cols, prefix, suffix, pattern)

    if method == 't':
        result = _one_sample_t(df, cols, null, alpha)
    
    elif method == 'sign':
        result = _one_sample_sign(df, cols, null, alpha, proportion = True)
    
    elif method == 'bootstrap':
        raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result    

def _one_sample_t(
    df: pd.DataFrame,
    cols: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample t test.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        null (float, optional): The population mean under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - 'mean_diff': The difference between observed and null means.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.    """

    desc = df[cols].agg(['count', 'mean'], axis = 0)

    result = scipy.stats.ttest_1samp(
        df[cols].to_numpy(),
        popmean = null,
        alternative = 'two-sided',
        nan_policy = 'omit'
    )
    
    return _create_test_frame(
        cols,
        desc.loc['mean'].to_numpy() - null,
        result.pvalue, # type: ignore
        desc.loc['count'].to_numpy(),
        alpha,
        'mean_diff',
    )

def _one_sample_sign(
    df: pd.DataFrame,
    cols: list[str],
    null: float,
    alpha: float,
    proportion: bool = False,
) -> pd.DataFrame:
    """Run a one-sample sign test.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        null (float, optional): The population median under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.
        proportion (bool): Whether the test is on a proportion or continuous 

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - A descriptive difference column, dynamically named based on the test.
                * 'median_diff' when `proportion = False`
                * 'prop_diff' when `proportion = True`
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    test_statistics = []
    p_values = []
    counts = []

    for col in cols:
        data = df[col].dropna()
        diffs = data - null
        positives = np.sum(diffs > 0)
        total_trials = len(data) if proportion else np.sum(diffs != 0)

        if total_trials == 0:
            test_statistics.append(np.nan)
            p_values.append(np.nan)
            counts.append(len(data))
        
        else:
            result = scipy.stats.binomtest(
                positives,
                total_trials,
                p = null if proportion else 0.5
            )

            test_statistic = np.mean(df[col]) - null if proportion else np.median(diffs)
            test_statistics.append(test_statistic)
            p_values.append(result.pvalue)
            counts.append(len(data))
    
    return _create_test_frame(
        cols,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        'prop_diff' if proportion else 'median_diff',
    )

def _one_sample_wilcoxon(
    df: pd.DataFrame,
    cols: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample Wilcoxon signed-rank test.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        null (float, optional): The population median under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - 'median_diff': The difference between observed and null medians.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    desc = df[cols].agg(['count', 'median'], axis = 0)

    result = scipy.stats.wilcoxon(
        df[cols].to_numpy() - null,
        alternative = 'two-sided',
        zero_method = 'wilcox',
        nan_policy = 'omit', # type: ignore
    )
    
    return _create_test_frame(
        cols,
        desc.loc['median'].to_numpy() - null,
        result.pvalue, # type: ignore
        desc.loc['count'].to_numpy(),
        alpha,
        'median_diff'
    )

def _create_test_frame(
    cols: list[str],
    test_statistics: np.ndarray,
    p_values: np.ndarray,
    counts: np.ndarray,
    alpha: float,
    statistic_name: str,
) -> pd.DataFrame:
    """Package test results into a DataFrame.

    Args:
        cols (list[str]): The labels associated with each column.
        test_statistics (np.ndarray): The array of test statistics.
        p_values (np.ndarray): The array of p values.
        counts (np.ndarray): The array of column non-nan counts.
        alpha (float): The desired alpha level.
        statistic_name (str): The name of the kind of values in `test_statistics`.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols` and columns `statistic_name`, 'p_value', 'stat_sig', 'count'.
    """
    
    data_dict = {
        f'{statistic_name}': test_statistics.astype(float),
        'p_value': p_values.astype(float),
        'stat_sig': (p_values < alpha).astype(bool),
        'count': counts.astype(int),
    }

    result = pd.DataFrame(
        data_dict,
        index = cols
    )

    return result  
  
# TODO: Add other test methods...
# test_independent(): independent t, mann-whitney u, one-way anova, kruskal-wallis
# test_dependent(): paired t, wilcoxon signed-rank, mcnemar asymptotic, mcnemar exact binomial
# test_regression(): linear, logistic

# TODO: Add p-value correction methods...bonferroni, holm-bonferroni, benjamini-hochberg

# TODO: Consider adding test of normality (and maybe leverage alongside sample size when method is unspecified in higher-level funcs?)

# TODO: Consider adding power functions
# If i want to add a power func, desc is df.agg_rows(.., ['count', 'std'])
# rng = np.random.default_rng(0)

# for col in desc.columns:
#     rvs = lambda size: rng.normal(
#         loc = desc[col].loc['alt_mean'],
#         scale = desc[col].loc['std'],
#         size = size,
#     )

#     power = scipy.stats.power(
#         scipy.stats.ttest_1samp,
#         rvs,
#         desc[col].loc['count'],
#         significance = alpha,
#         n_resamples = 2000,
#         kwargs = {'popmean': 0}
#     )

#     desc.loc['power', col] = power.power