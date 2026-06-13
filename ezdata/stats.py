import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import proportion_confint
from typing import Callable
import re
from . import prep
from .selector import Selector

def agg_cols(
    df: pd.DataFrame,
    method: str,
    target_col: str,
    *,
    drop_inputs: bool = False,
    cols: list[str] | set[str] | str | Selector | None = None,
) -> pd.DataFrame:
    """Aggregate across columns to create a new column.

    Applies an aggregation function given by `method` across columns (i.e., axis = 1). 

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The aggregation method. Supported choices: 'min', 'max', 'sum', 'mean', 'median', 'count', 'std', 'var', 'prod', 'or', 'and'.
        target_col (str): The target column name in which to store the aggregated values.
        drop_inputs (bool): If true, drops the columns used in aggregation (i.e., those indicated the column-selection parameters). Defaults to False.
        cols (list[str] | set[str] | str | Selector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

    Raises:
        ValueError: If the aggregation method specified in `method` isn't recognized.

    Returns:
        pd.DataFrame: The DataFrame with the aggregated values stored in the target column.
    """
    
    df = df.copy()
    cols = prep._resolve_selection(df, cols)

    valid_methods = {'min', 'max', 'sum', 'mean', 'median', 'count', 'nunique', 'std', 'var', 'prod', 'or', 'and'}
    pd_methods = {'min', 'max', 'sum', 'mean', 'median', 'count', 'nunique', 'std', 'var', 'prod'}

    if method in pd_methods:
        df[target_col] = df[cols].agg(method, axis = 1)

    elif method == 'or':
        df[target_col] = df[cols].astype(bool).any(axis = 1).astype(int)

    elif method == 'and':
        df[target_col] = df[cols].astype(bool).all(axis = 1).astype(int)

    else:
        raise ValueError(f'Unrecognized aggregation method \'{method}\'. Supported choices: {valid_methods}')

    if drop_inputs:
        cols_to_drop = [col for col in cols if col != target_col]
        df = df.drop(columns = cols_to_drop)
    
    return df

def agg_rows(
    df: pd.DataFrame,
    method: str | list[str],
    *,
    cols: list[str] | set[str] | str | Selector | None = None,
) -> pd.DataFrame | pd.Series:
    """Aggregate across rows of the given DataFrame to create a new DataFrame or Series.

    Applies aggregation function(s) given by 'method' across rows (i.e., axis = 0). 

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str | list[str]): The aggregation method. Supported choices: 'min', 'max', 'sum', 'mean', 'median', 'count', 'std', 'var', 'prod'.
        cols (list[str] | set[str] | str | Selector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

    Raises:
        ValueError: If a given aggregation method in `method` isn't recognized.

    Returns:
        pd.DataFrame | pd.Series: The resulting DataFrame or Series.
    """

    df = df.copy()
    cols = prep._resolve_selection(df, cols)

    valid_methods = {'min', 'max', 'sum', 'mean', 'median', 'count', 'nunique', 'std', 'var', 'prod'}
    method_map = {}

    if isinstance(method, list):
        for val in method:
            if val not in valid_methods:
                raise ValueError(
                    f'Unrecognized aggregation method \'{method}\'. Supported choices: {valid_methods}.'
                )
            
    elif isinstance(method, str):
        if method not in valid_methods:
            raise ValueError(
                f'Unrecognized aggregation method \'{method}\'. Supported choices: {valid_methods}.'
            )
        
    for col in cols:
        method_map[col] = method     

    return df[cols].agg(method_map, axis = 0)    

def calc_ci(
    df: pd.DataFrame,
    method: str,
    *,
    alpha: float = 0.05,
    random_state: int = 0,
    metric: str = 'mean',
    cols: list[str] | set[str] | str | Selector | None = None,
) -> pd.DataFrame:
    """Calculate confidence intervals according to the given method.
    
    Also includes associated statistics used in the calcualtion.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str, optional): The CI-calculation method. Supported choices: 
            * Parametric: 'z', 't'
            * Proportion: 'wald', 'wilson', 'agresti_coull', 'clopper_pearson' (or 'beta'), 'jeffreys'
            * Bootstrap: 'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        random_state (int, optional): A random-number-generator seed, relevant for bootstrapping. Defaults to 0.
        metric (str, optional): The measure of central tendency to craft the interval around, relevant for bootstrapping. Supported choices: 'mean', 'median'. Defaults to 'mean'.
        cols (list[str] | set[str] | str | Selector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.
        
    Returns:
        pd.DataFrame: A DataFrame with indices matching the columns specified in the column-selection parameters and columns 'point_estimate', 'lower', 'upper', 'count'.
    """
    
    df = df.copy()
    cols = prep._resolve_selection(df, cols)
    
    parametric_methods = {'z', 't'}
    proportion_methods = {'wald', 'wilson', 'agresti_coull', 'clopper_pearson', 'beta', 'jeffreys'}
    bootstrap_methods = {'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'}

    method, alpha = _validate_ci_args(
        method,
        alpha, 
        parametric_methods | proportion_methods | bootstrap_methods
    )

    if method in parametric_methods:
        result = _calc_ci_parametric(df, cols, alpha, method)

    elif method in proportion_methods:
        result = _calc_ci_proportion(df, cols, alpha, method)

    elif method in bootstrap_methods:
        result = _calc_ci_bootstrap(df, cols, alpha, method, random_state, metric)

    return result

def _validate_ci_args(
    method: str,
    alpha: float,
    valid_methods: set[str],
) -> tuple[str, float]:
    """Validate the arguments given to compute_ci().

    Additionally converts the string argument to `method` to lowercase before validating.

    Args:
        method (str): The desired CI-calculation method.
        alpha (float): The desired alpha.
        valid_methods (set[str]): A set of valid methods.

    Raises:
        ValueError: If string argument for `method` isn't recognized.
        ValueError: If float argument for `alpha` isn't between 0 and 1 exclusive.

    Returns:
        tuple[str, float]: A tuple containing lowercase `method` and `alpha`.
    """

    method = method.lower()

    if method not in valid_methods:
        raise ValueError(f'Unrecognized argument for method: \'{method}\'. Supported choices: {valid_methods}.')
    
    if not (0 < alpha < 1):
        raise ValueError(
            f'Invalid argument for alpha: \'{alpha}\'. Values should be greater than 0 '
            'and less than 1. Typical values might be in the range of 0.01 to 0.1.'
        )

    return method, alpha

def _create_ci_frame(
    cols: list[str],
    point_estimates: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    counts: np.ndarray,
) -> pd.DataFrame:
    """Package CI-calculation results into a DataFrame.

    Args:
        cols (list[str]): The labels associated with each column.
        point_estimates (np.ndarray): The array of point estimates.
        lower_bounds (np.ndarray): The array of lower bounds.
        upper_bounds (np.ndarray): The array of upper bounds.
        counts (np.ndarray): The array of column non-nan counts.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols` and columns 'point_estimate', 'lower', 'upper', 'count'.
    """
    
    data_dict = {
        'point_estimate': point_estimates.astype(float),
        'lower': lower_bounds.astype(float),
        'upper': upper_bounds.astype(float),
        'count': counts.astype(int),
    }

    result = pd.DataFrame(
        data = data_dict,
        index = cols,
    )

    return result    

def _calc_ci_parametric(
    df: pd.DataFrame,
    cols: list[str],
    alpha: float,
    distribution: str,
) -> pd.DataFrame:
    """Calculate parametric confidence intervals.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        alpha (float): The desired alpha.
        distribution (str): The distribution to use. Supported choices: 'z', 't'.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols` and columns 'point_estimate', 'lower', 'upper', 'count'.
    """

    stats = df[cols].agg(['mean', 'std', 'count'], axis = 0)
    standard_error = stats.loc['std'] / np.sqrt(stats.loc['count'])

    if distribution == 'z':
        lower, upper = scipy.stats.norm.interval(
            confidence = 1 - alpha,
            loc = stats.loc['mean'],
            scale = standard_error,
        )

    elif distribution == 't':
        lower, upper = scipy.stats.t.interval(
            confidence = 1 - alpha,
            df = stats.loc['count'] - 1, 
            loc = stats.loc['mean'],
            scale = standard_error,
        )


    return _create_ci_frame(
        cols, 
        stats.loc['mean'].to_numpy(),
        np.array(lower),
        np.array(upper),
        stats.loc['count'].to_numpy(),
    )

def _calc_ci_proportion(
    df: pd.DataFrame,
    cols: list[str],
    alpha: float,
    method: str,
) -> pd.DataFrame:
    """Calculate proportion confidence intervals.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        alpha (float): The desired alpha.
        method (str): The CI-calculation method. Supported choices: 'wald', 'wilson', 'agresti_coull', 'clopper_pearson' (or 'beta'), 'jeffreys'.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols` and columns 'point_estimate', 'lower', 'upper', 'count'.
    """

    sm_mapping = {
        'wald': 'normal',
        'wilson': 'wilson',
        'agresti_coull': 'agresti_coull',
        'clopper_pearson': 'beta',
        'beta': 'beta',
        'jeffreys': 'jeffreys'
    }
    
    stats = df[cols].agg(['mean', 'count', 'sum'], axis = 0)
    
    lower, upper = proportion_confint(
        count = stats.loc['sum'].to_numpy(),
        nobs = stats.loc['count'].to_numpy(),
        alpha = alpha,
        method = sm_mapping[method],    
    )

    return _create_ci_frame(
        cols,
        stats.loc['mean'].to_numpy(),
        np.array(lower),
        np.array(upper),
        stats.loc['count'].to_numpy(),
    )

def _calc_ci_bootstrap(
    df: pd.DataFrame,
    cols: list[str],
    alpha: float,
    method: str,
    random_state: int,
    metric: str,
) -> pd.DataFrame:
    """Calculate bootstrap confidence intervals.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        alpha (float): The desired alpha.
        method (str): The CI-calculation method. Supported choices: 'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'.
        random_state (int): A random-number-generator seed.
        metric (str): The measure of central tendency to craft the interval around. Supported choices: 'mean', 'median'.
        
    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols` and columns 'point_estimate', 'lower', 'upper', 'count'.
    """

    sp_mapping = {
        'bootstrap_percentile': 'percentile',
        'bootstrap_basic': 'basic',
        'bootstrap_bca': 'BCa',
    }

    point_estimates = []
    lowers = []
    uppers = []
    counts = []

    metric, metric_func = _get_metric_bootstrap(metric)
    stats = df[cols].agg([f'{metric}', 'count', 'sum'], axis = 0)
    
    for col in cols:
        data = df[col].dropna().to_numpy()
        point_estimates.append(stats[col].loc[f'{metric}'])
        counts.append(stats[col].loc['count'])

        if len(data) < 5:
            lowers.append(np.nan)
            uppers.append(np.nan)
            continue
            
        result = scipy.stats.bootstrap(
            data = (data, ),
            statistic = metric_func,
            confidence_level = 1 - alpha,
            method = sp_mapping[method],
            n_resamples = 2000,
            random_state = random_state, # type: ignore
        )

        lowers.append(result.confidence_interval.low)
        uppers.append(result.confidence_interval.high)

    return _create_ci_frame(
        cols,
        np.array(point_estimates),
        np.array(lowers),
        np.array(uppers),
        np.array(counts),
    )

def _get_metric_bootstrap(
    metric: str,
) -> tuple[str, Callable[[np.ndarray], float]]:
    
    metric = metric.lower()
    
    if metric == 'mean':
        metric_func = np.mean
    
    elif metric == 'median':
        metric_func = np.median

    else:
        raise ValueError(
            f'Metric argument \'{metric}\' isn\'t recognized. '
            'Supported choices: \'mean\', \'median\'.'
        )
    
    return metric, metric_func