import pandas as pd
import warnings
import re

_FIX_CHAR_MAP = str.maketrans({
    '’': "'",
    '‘': "'",
    '“': '"',
    '”': '"',
    '\xa0': ' ',
    '–': '-',
    '−': '-',
    '\u00A0': ' ',
    '\u2013': '-',
    '\u2014': '-',
    '\u202f': ' '
})

PATTERN_ALIDA_OTHER_OE = re.compile(r'other.*_0$', re.IGNORECASE)
PATTERN_CAPTURE_LEADING_INT = re.compile(r'^(\d+)', re.IGNORECASE)
_PATTERN_BIN_METHOD = re.compile(r'(?P<kind>i|q)(?P<number>\d+\.?\d*)', re.IGNORECASE)

def remove_cols(
    df: pd.DataFrame, 
    *,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Remove columns whose labels match the given criteria.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str] | set[str] | str | None, optional): Column(s) to remove. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns to remove. Defaults to None.
        suffix (str | None, optional): The suffix of columns to remove. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns to remove. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The DataFrame with columns removed.
    """
    
    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    df = df.drop(columns = cols)

    return df

def rename_cols(
    df: pd.DataFrame,
    mapper: dict | str | re.Pattern,
    *,
    regex_keys: bool = False,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Rename DataFrame columns according to the given mapper.

    Args:
        df (pd.DataFrame): The DataFrame.
        mapper (dict | str | re.Pattern): A dictionary mapping existing column names to desired column names. Alternatively, this can be a regex pattern with a single capture group to define what to extract from exisiting column names.
        regex_keys (bool, optional): Whether to treat string keys of `mapper` as regex patterns. Defaults to False.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    df = _recode(
        df, 
        cols,
        'columns', 
        mapper, 
        regex_keys = regex_keys,
    )
    
    return df

def recode_vals(
    df: pd.DataFrame,
    mapper: dict | str | re.Pattern,
    *,
    new_col_prefix: str | None = None,
    regex_keys: bool = False,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Recode DataFrame values according to the given mapper.

    Args:
        df (pd.DataFrame): The DataFrame.
        mapper (dict | str | re.Pattern): A dictionary mapping existing values to desired values. Alternatively, this can be a regex pattern with a single capture group to define what to extract from exisiting values.
        new_col_prefix (str | None, optional): A prefix to add to new columns with the potentially-recoded values. If None, will not create new columns. Defaults to None.
        regex_keys (bool, optional): Whether to treat string keys of `mapper` as regex patterns. Defaults to False.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    df = _recode(
        df, 
        cols,
        'values', 
        mapper, 
        new_col_prefix = new_col_prefix,
        regex_keys = regex_keys,
    )
    
    return df

def clean_df(
    df: pd.DataFrame,
    *,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Standardize characters and strip strings in DataFrame.

    Applies character translation and stripping to column names and values of object or string columns.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str] | set[str] | str | None, optional): Column(s) to clean. If None, cleans all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns to clean. Defaults to None.
        suffix (str | None, optional): The suffix of columns to clean. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns to clean. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The DataFrame with standardized characters.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    rename_dict = {col: str(col).translate(_FIX_CHAR_MAP).strip() for col in cols}
    df = df.rename(columns = rename_dict)

    updated_cols = rename_dict.values()
    str_cols = df[updated_cols].select_dtypes(include=['object', 'string']).columns
    
    for col in str_cols:
        df[col] = df[col].str.translate(_FIX_CHAR_MAP).str.strip()

    return df

def clean_arg(
    arg: str | list | dict,
) -> str | list | dict:
    """Standardize characters and strip strings in argument.

    Args:
        arg (str | list | dict): The string, list, or dict to standardize.

    Returns:
        str | list | dict: The standardized argument.
    """

    if isinstance(arg, str):
        return arg.translate(_FIX_CHAR_MAP).strip()
    
    elif isinstance(arg, list):
        return [clean_arg(val) for val in arg]
    
    elif isinstance(arg, dict):
        return {clean_arg(k): clean_arg(v) for k, v in arg.items()}
    
    return arg

def remove_verbal_anchors(
    df: pd.DataFrame,
    *,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Extract leading digits from string values in a DataFrame.

    Takes leading digits from values in string or object columns and coerces the columns to numeric.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The updated DataFrame.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    str_cols = list(df[cols].select_dtypes(include=['object', 'string']).columns)

    df = _recode(
        df,
        str_cols,
        'values',
        PATTERN_CAPTURE_LEADING_INT,
    )

    for col in str_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    return df

def filter_straightliners(
    df: pd.DataFrame,
    *,
    min_unique: int = 2,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Filter DataFrame values based on the required minimum number of unique values in a row.

    Replaces all values in a row that fails to meet the requirement with NaN.

    Args:
        df (pd.DataFrame): The DataFrame.
        min_unique (int, optional): The minimum number of unique values desired in a row. If below this number, values will be replaced with NaN. Defaults to 2.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The DataFrame with straightliners' values replaced.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    df[cols] = df[cols].where(df[cols].nunique(axis = 1) >= min_unique)

    return df

def bin(
    df: pd.DataFrame,
    method: str | list,
    *,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Bin DataFrame values.

    Bins on the basis of number of quantiles, number of equal-width intervals, or explicitly given edges.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str | list): The binning method to apply.
            * A string of the form 'q#': Quantile binning (e.g., 'q4' and 'q2' bins on the basis of quartiles or a median split, respectively).
            * A string of the form 'i#': Interval binning (e.g., 'i5' will create 5 equal-width intervals that capture the range of values).
            * A list of numbers: Explicitly defined bin edges.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The binned DataFrame.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    if isinstance(method, str):
        df = _bin_by_string(df, method, cols = cols)
        
    elif isinstance(method, list):
        df = _bin_by_edges(df, method , cols = cols)

    return df

def filter_by_bounds(
    df: pd.DataFrame,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Filter DataFrame values based on the given minimum and maximum.

    Replaces values that exceed the bounds with NaN.

    Args:
        df (pd.DataFrame): The DataFrame.
        min (int, optional): The minimum value to keep. Defaults to None.
        max (int, optional): The maximum value to keep. Defaults to None.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    if min_val is not None:
        df[cols] = df[cols].where(df[cols] >= min_val)

    if max_val is not None:
        df[cols] = df[cols].where(df[cols] <= max_val)

    return df

def filter_by_iqr(
    df: pd.DataFrame,
    *,
    factor: float | int = 1.5,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | re.Pattern | None = None,
) -> pd.DataFrame:
    """Filter DataFrame values based on the IQR method.

    Replaces values that exceed the calculated bounds with NaN. Defines a maximum value as Q3 + factor * IQR and a minimum value as Q1 - `factor` * IQR where IQR = Q3 - Q1.

    Args:
        df (pd.DataFrame): The DataFrame.
        factor (float | int, optional): The factor by which to multiply the IQR to determine min and max values. Defaults to 1.5.
        cols (list[str] | set[str] | str | None, optional): Column(s) on which to operate. If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
        suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
        pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to operate. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    df, cols = _prep_args(df, cols, prefix, suffix, pattern)

    qs = df[cols].quantile([0.25, 0.75], axis = 0)
    iqrs = qs.loc[0.75] - qs.loc[0.25]

    mins = qs.loc[0.25] - iqrs * factor
    maxes = qs.loc[0.75] + iqrs * factor

    df[cols] = df[cols].where((df[cols] <= maxes) & (df[cols] >= mins))

    return df

def _bin_by_string(
    df: pd.DataFrame,
    method: str,
    cols: list[str],
) -> pd.DataFrame:
    """Bin DataFrame values based on the given string.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): String representing the desired binning method, of the form 'q#' or 'i#'.
        cols (list[str]): A list of columns on which to operate. 

    Raises:
        ValueError: If string argument for `method` doesn't follow the expected pattern.

    Returns:
        pd.DataFrame: The binned DataFrame.
    """

    method_match = re.match(_PATTERN_BIN_METHOD, method)

    if not method_match: 
        raise ValueError(
            f'String argument \'{method}\' for parameter \'method\' doesn\'t follow '
            'the expected pattern: \'q#\' or \'i#\'.'
        )
    
    method_kind = method_match.group('kind')
    method_number = float(method_match.group('number'))

    if method_number.is_integer():
        method_number = int(method_number)

    elif isinstance(method_number, float):
        method_number = int(method_number)
        warnings.warn(
            f'String argument for parameter \'method\' included a float. \'{method}\' '
            'was converted to \'{method_kind}{method_number}\''
        )
    
    for col in cols:
        if method_kind == 'q':
            df[col] = pd.qcut(df[col], method_number).astype(str)

        elif method_kind == 'i':
            df[col] = pd.cut(df[col], method_number).astype(str)

    return df

def _bin_by_edges(
    df: pd.DataFrame,
    method: list[int | float],
    cols: list[str],
) -> pd.DataFrame:
    """Bin DataFrame values based on the given edges.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (list[int | float]): List of bin edges.
        cols (list[str]): A list of columns on which to operate. 

    Returns:
        pd.DataFrame: The binned DataFrame.
    """

    for col in cols:
        df[col] = pd.cut(df[col], method).astype(str)

    return df

def _prep_args(
    df: pd.DataFrame,
    cols: list[str] | set[str] | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: re.Pattern | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Make a copy of the DataFrame and get a list of selected column names.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str] | set[str] | None, optional): Desired column name(s). If None, includes all columns. Defaults to None.
        prefix (str | None, optional): The prefix of desired column names. Defaults to None.
        suffix (str | None, optional): The suffix of desired column names. Defaults to None.
        pattern (re.Pattern | str | None, optional): A regex pattern describing desired column names. Defaults to None.

    Note:
        Selection parameters (e.g., `cols`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple of the DataFrame copy and the list of desired column names.
    """
            
    df = df.copy()
    all_cols = df.columns.tolist()

    if cols is None and prefix is None and suffix is None and pattern is None:
        return df, all_cols

    if cols is None:
        cols = set(all_cols)

    elif isinstance(cols, str):
        cols = set([cols])

    elif isinstance(cols, list):
        cols = set(cols)     

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    matched_cols = [
        col for col in all_cols
        if (cols is None or col in cols)
        and (prefix is None or col.startswith(prefix))
        and (suffix is None or col.endswith(suffix))
        and (pattern is None or re.search(pattern, col))
    ]
    
    return df, matched_cols

def _recode(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
    mapper: dict | str | re.Pattern,
    *,
    regex_keys: bool = False,
    new_col_prefix: str | None = None,
) -> pd.DataFrame:
    """Recode column names or values.

    Helper function used across multiple public functions (e.g., rename_cols, recode_vals).

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        target (str): The target of recoding. Supported choices: 'columns', 'values'.
        mapper (dict | str | re.Pattern): A dictionary mapping existing values to desired values. Alternatively, this can be a regex pattern with a single capture group to define what to extract from exisiting values.
        regex_keys (bool): Whether to treat string keys of `mapper` as regex patterns. Defaults to False.
        new_col_prefix (str | None, optional): A prefix to add to new columns with the potentially-recoded values. If None, will not create new columns. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with recoded columns or values.
    """

    mapper = _standardize_mapper(df, cols, mapper, target, regex_keys)

    if new_col_prefix is not None:
        dest_cols = [new_col_prefix + col for col in cols]

    else:
        dest_cols = cols

    if target == 'columns':
        df = df.rename(columns = mapper)

    elif target == 'values':
        df[dest_cols] = df[cols].replace(mapper).to_numpy()
    
    return df

def _standardize_mapper(
    df: pd.DataFrame,
    cols: list[str],
    mapper: dict | str | re.Pattern,
    target: str,
    regex_keys: bool, 
):
    """Standardize a mapper argument.

    Helper function to _recode.

    Args:
        df (pd.DataFrame): The DataFrame
        cols (list[str]): The columns on which to operate.
        mapper (dict | str | re.Pattern): A dictionary mapping existing values to desired values. Alternatively, this can be a regex pattern with a single capture group to define what to extract from exisiting values.
        target (str): The target of recoding. Supported choices: 'columns', 'values'.
        regex_keys (bool): Whether to treat string keys of `mapper` as regex patterns.

    Raises:
        ValueError: If `target` is an unsupported value.
        ValueError: If `mapper` is an unsupported type.

    Returns:
        dict: The standardized mapper.
    """
    
    if target == 'columns':
        values_to_map = set(cols)

    elif target == 'values':
        values_to_map = set()
        
        for col in cols:
            values_to_map.update(df[col].unique())

    else:
        raise ValueError(
            f'Mapping `target` \'{target}\' isn\'t supported. '
            'Supported choices: \'columns\', \'values\'.'
        )
    
    values_to_map = list(values_to_map)
    
    if isinstance(mapper, dict):
        mapper = _standardize_dict_mapper(mapper, values_to_map, regex_keys)

    elif isinstance(mapper, str) or isinstance(mapper, re.Pattern):
        mapper = _standardize_extract_mapper(mapper, values_to_map)

    else:
        raise ValueError(
            f'`mapper` must be a string, compiled regex pattern, or a dictionary. '
            f'Received {type(mapper)}'
        )
    
    return mapper   

def _standardize_dict_mapper(
    mapper: dict,
    values_to_map: list,
    regex_keys: bool,
) -> dict:
    """Standardize dictionary mapper argument.

    Helper function to _standardize_mapper.

    Args:
        mapper (dict): A dictionary mapping existing values and/or regex patterns to new values.
        values_to_map (list): A list of values from which to create a map.
        regex_keys (bool): Whether to treat string keys of `mapper` as regex patterns.

    Returns:
        dict: The updated mapper.
    """
      
    new_mapper = {}
    
    for k, v in mapper.items():
        if isinstance(k, re.Pattern) or regex_keys == True:
            if isinstance(k, str):
                k = re.compile(k)

            for val in values_to_map:
                if re.search(k, str(val)):
                    new_mapper[val] = v

        else:
            new_mapper[k] = v
            
    return new_mapper

def _standardize_extract_mapper(
    mapper: str | re.Pattern,
    values_to_map: list
) -> dict:
    """Standardize string or regex pattern mapper argument.

    Helper function to _standardize_mapper.

    Args:
        mapper (str | re.Pattern): A regex pattern with a single capture group to define what to extract from exisiting values
        values_to_map (list): A list of values from which to create a map.

    Raises:
        ValueError: If `mapper` has less than or greater than 1 capture group.

    Returns:
        dict: The updated mapper.
    """
    
    if isinstance(mapper, str):
        mapper = re.compile(mapper)

    n_capture_groups = mapper.groups

    if n_capture_groups != 1:
        raise ValueError(
            f'Regex pattern `mapper` had {n_capture_groups} capture groups '
            'but exactly 1 is required to determine what to extract.'
        )
    
    new_mapper = {}

    for val in values_to_map:
        match = re.search(mapper, str(val))

        if match:
            new_mapper[val] = match.group(1)

    return new_mapper    