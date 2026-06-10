import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import proportion_confint
import re
import warnings
import sys

class DataProcessor:
    """Create a DataProcessor object to prepare data for plotting.
    """

    FIX_CHAR_MAP = str.maketrans({
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
    PATTERN_LEADING_INT = re.compile(r'^(\d+)', re.IGNORECASE)
    PATTERN_BIN_METHOD = re.compile(r'(?P<kind>i|q)(?P<number>\d+\.?\d*)', re.IGNORECASE)

    def _prep_args(
        self, 
        df: pd.DataFrame, 
        cols: list[str] | set[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> tuple[pd.DataFrame, list]:
        """Prepare df and cols arguments.

        Helper method to handle DataFrame copying and column selection.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | str | None, optional): Column name(s).
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): A column-name prefix. Defaults to None.
            suffix (str | None, optional): A column-name suffix. Defaults to None.
            pattern (str | re.Pattern | None, optional): A column-name regex pattern. 
                Defaults to None.

        Returns:
            tuple[pd.DataFrame, list[str]]: A tuple containing a copy of the DataFrame and the 
                list of column names.
        """
        
        df = df.copy()

        if cols is None and prefix is None and suffix is None and pattern is None:
            return df, df.columns.tolist()

        if isinstance(cols, str):
            cols = [cols]

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        matched_cols = self._get_cols(df, cols = cols, prefix = prefix, suffix = suffix, pattern = pattern)

        return df, matched_cols
    
    def remove_cols(
        self, 
        df: pd.DataFrame, 
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Remove columns whose labels match the given criteria.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | str | None, optional): Column(s) to remove.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns to remove. Defaults to None.
            suffix (str | None, optional): The suffix of columns to remove. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to 
                remove. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The DataFrame with columns removed.
        """
        
        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        df = df.drop(columns = cols)

        return df
    
    def fix_characters_df(
        self,
        df: pd.DataFrame,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Standardize characters and strip strings in DataFrame.

        Applies character translation and stripping to column names and values of object or string columns.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | str | None, optional): Column(s) to clean.
                If None, cleans all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns to clean. Defaults to None.
            suffix (str | None, optional): The suffix of columns to clean. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to 
                clean. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The DataFrame with standardized characters.
        """

        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        rename_dict = {col: str(col).translate(self.FIX_CHAR_MAP).strip() for col in cols}
        df = df.rename(columns = rename_dict)

        updated_cols = rename_dict.values()
        str_cols = df[updated_cols].select_dtypes(include=['object', 'string']).columns
        
        for col in str_cols:
            df[col] = df[col].str.translate(self.FIX_CHAR_MAP).str.strip()

        return df
    
    def fix_characters_arg(
        self, 
        arg: str | list | dict,
    ) -> str | list | dict:
        """Standardize characters and strip strings in argument.

        Args:
            arg (str | list | dict): The string, list, or dict to standardize.

        Returns:
            str | list | dict: The standardized argument.
        """

        if isinstance(arg, str):
            return arg.translate(self.FIX_CHAR_MAP).strip()
        
        elif isinstance(arg, list):
            return [self.fix_characters_arg(val) for val in arg]
        
        elif isinstance(arg, dict):
            return {self.fix_characters_arg(k): self.fix_characters_arg(v) for k, v in arg.items()}
        
        return arg

    def remove_verbal_anchors(
        self,
        df: pd.DataFrame,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Extract leading digits from string values in a DataFrame.

        Takes leading digits from values in string or object columns and coerces the columns to numeric.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | str | None, optional): Column(s) on which to operate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
            suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to 
                operate. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The updated DataFrame.
        """

        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        str_cols = df[cols].select_dtypes(include=['object', 'string']).columns

        for col in str_cols:
            df[col] = df[col].astype(str).str.extract(self.PATTERN_LEADING_INT)[0]
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        return df
    
    def filter_straightliners(
        self,
        df: pd.DataFrame,
        min_unique: int = 2,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Filter DataFrame values based on the required minimum number of unique values in a row.

        Replaces all values in a row that fails to meet the requirement with NaN.

        Args:
            df (pd.DataFrame): The DataFrame.
            min_unique (int, optional): The minimum number of unique values desired in a row. 
                If below this number, values will be replaced with NaN. Defaults to 2.
            cols (list[str] | str | None, optional): Column(s) on which to operate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
            suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to 
                operate. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The DataFrame with straightliners' values replaced.
        """

        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        df[cols] = df[cols].where(df[cols].nunique(axis = 1) >= min_unique)

        return df
    
    def bin(
        self,
        df: pd.DataFrame, 
        method: str | list,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Bin DataFrame values.

        Bins on the basis of number of quantiles, number of equal-width intervals, or explicitly given edges.

        Args:
            df (pd.DataFrame): The DataFrame.
            method (str | list): The binning method to apply. Options include:
                * A string of the form 'q#': Quantile binning (e.g., 'q4' and 'q2' bins on the basis of 
                quartiles or a median split, respectively).
                * A string of the form 'i#': Interval binning (e.g., 'i5' will create 5 equal-width 
                intervals that capture the range of values).
                * A list of numbers: Explicitly defined bin edges.
            cols (list[str] | str | None, optional): Column(s) on which to operate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
            suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which to 
                operate. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The binned DataFrame.
        """

        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        if isinstance(method, str):
            df = self._bin_by_string(df, method, cols = cols)
            
        elif isinstance(method, list):
            df = self._bin_by_edges(df, method , cols = cols)

        return df
    
    def _bin_by_string(
        self,
        df: pd.DataFrame,
        method: str,
        cols: list[str],
    ) -> pd.DataFrame:
        """Bin DataFrame values based on the given string.

        Helper method to handle str input for 'method' arg of bin().

        Args:
            df (pd.DataFrame): The DataFrame.
            method (str): String representing the desired binning method, of the form 'q#' or 'i#'.
            cols (list[str]): A list of columns on which to operate. 

        Raises:
            ValueError: If string argument for 'method' doesn't follow the expected pattern.

        Returns:
            pd.DataFrame: The binned DataFrame.
        """

        method_match = re.match(self.PATTERN_BIN_METHOD, method)

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
        self,
        df: pd.DataFrame,
        method: list[int | float],
        cols: list[str],
    ) -> pd.DataFrame:
        """Bin DataFrame values based on the given edges.

        Helper method to handle list input for 'method' arg of bin().

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

    
    def filter_by_bounds(
        self,
        df: pd.DataFrame,
        min_val: int | None = None,
        max_val: int | None = None,
        cols: list[str] | str | None = None,
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
            cols (list[str] | str | None, optional): Column(s) on which to operate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
            suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which 
                to operate. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """

        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        if min_val is not None:
            df[cols] = df[cols].where(df[cols] >= min_val)

        if max_val is not None:
            df[cols] = df[cols].where(df[cols] <= max_val)

        return df

    def filter_by_iqr(
        self,
        df: pd.DataFrame,
        factor: float | int = 1.5,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Filter DataFrame values based on the IQR method.

        Replaces values that exceed the calculated bounds with NaN.
        
        Defines a maximum value as Q3 + factor * IQR and a minimum value as Q1 - factor * IQR 
        where IQR = Q3 - Q1.

        Args:
            df (pd.DataFrame): The DataFrame.
            factor (float | int, optional): The factor by which to multiply the IQR to determine 
                min and max values. Defaults to 1.5.
            cols (list[str] | str | None, optional): Column(s) on which to operate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
            suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which 
                to operate. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """

        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        qs = df[cols].quantile([0.25, 0.75], axis = 0)
        iqrs = qs.loc[0.75] - qs.loc[0.25]

        mins = qs.loc[0.25] - iqrs * factor
        maxes = qs.loc[0.75] + iqrs * factor

        df[cols] = df[cols].where((df[cols] <= maxes) & (df[cols] >= mins))

        return df
    
    def _validate_one_arg_used(
        self,
        **kwargs,  
    ) -> None:
        """Validate that only a single argument is not None.

        Raises:
            ValueError: If all are None.

        Returns:
            None: None.
        """

        class_name = self.__class__.__name__
        caller_name = sys._getframe(1).f_code.co_name

        param_list = []
        arg_list = []
        
        for param, arg in kwargs.items():
            param_list.append(param)
            
            if arg is not None: 
                arg_list.append(arg)

        if len(arg_list) > 1:
            warnings.warn(
                f'Only one of the following parameters can be used in {class_name}.{caller_name}: '
                '{param_list}. Defaulting to the documented precedence.'
            )

        elif len(arg_list) < 1:
            raise ValueError(
                f'One of the following parameters must be used in {class_name}.{caller_name}: '
                '{param_list}. All had a value of None.'
            )

        return None
    
    def _get_cols(
        self,
        df: pd.DataFrame,
        cols: list[str] | set[str] | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | None = None,
    ) -> list[str]:
        """Get a list of column names from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | set[str] | None, optional): Desired column name(s).
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of desired column names. Defaults to None.
            suffix (str | None, optional): The suffix of desired column names. Defaults to None.
            pattern (re.Pattern | None, optional): A regex pattern describing desired column 
                names. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            list[str]: The list of desired column names.
        """
                
        all_cols = df.columns.tolist()

        if cols is None and prefix is None and suffix is None and pattern is None:
            return all_cols

        if cols is None:
            cols = set(all_cols)

        elif isinstance(cols, list):
            cols = set(cols)        

        matched_cols = [
            col for col in all_cols
            if (cols is None or col in cols)
            and (prefix is None or col.startswith(prefix))
            and (suffix is None or col.endswith(suffix))
            and (pattern is None or re.search(pattern, col))
        ]
        
        return matched_cols
    
    def _stringify_mapper(
        self,
        mapper: dict[str | re.Pattern, str],
        regex_keys: bool,
        cols: list[str],
    ) -> dict[str, str]:
        """Standardize mapper argument to a dictionary of strings.

        Args:
            mapper (dict[str | re.Pattern, str]): A dictionary mapping strings and/or regex 
                patterns to strings.
            regex_keys (bool): Whether to treat string keys of mapper as regex patterns.
            cols (list[str]): A list of column names on which to operate.

        Returns:
            dict[str, str]: The updated mapper.
        """
        
        new_mapper = {}
        
        for k, v in mapper.items():

            if isinstance(k, re.Pattern) or regex_keys == True:
                if isinstance(k, str):
                    k = re.compile(k)

                for col in cols:
                    if re.search(k, col):
                        new_mapper[col] = v

            else:
                new_mapper[k] = v
                
        return new_mapper      
    
    def rename_cols(
        self,
        df: pd.DataFrame,
        mapper: dict,
        regex_keys: bool = False,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Rename DataFrame columns according to the given mapper.

        Args:
            df (pd.DataFrame): The DataFrame.
            mapper (dict[str | re.Pattern, str]): A dictionary mapping existing column names to 
                desired column names. Existing column names can take the form of strings and/or 
                compiled regex patterns.
            regex_keys (bool, optional): Whether to treat string keys of mapper as regex patterns.
                Defaults to False.
            cols (list[str] | str | None, optional): Column(s) on which to operate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
            suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which 
                to operate. Defaults to None.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The DataFrame with renamed columns.
        """
        
        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)
        mapper = self._stringify_mapper(mapper, regex_keys, cols)

        df = df.rename(columns = mapper)
        
        return df
    
    def agg_cols(
        self,
        df: pd.DataFrame,
        method: str,
        target_col: str,
        drop_inputs: bool = False,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame:
        """Aggregate across columns to create a new column.

        Applies an aggregation function given by 'method' across columns (i.e., axis = 1). 

        Args:
            df (pd.DataFrame): The DataFrame.
            method (str): The aggregation method. Supported choices: 'min', 'max', 'sum', 'mean', 'median',
                'count', 'std', 'var', 'prod', 'or', 'and'.
            target_col (str): The target column name in which to store the aggregated values.
            drop_inputs (bool): If true, drops the columns used in aggregation (i.e., those indicated 
                by 'cols'). Defaults to False.
            cols (list[str] | str | None, optional): Column(s) to aggregate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns to aggregate. Defaults to None.
            suffix (str | None, optional): The suffix of columns to aggregate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns 
                to aggregate. Defaults to None.

        Raises:
            ValueError: If the aggregation method specified in 'method' isn't recognized.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame: The DataFrame with the aggregated values stored in the target column.
        """
        
        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

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
        self,
        df: pd.DataFrame,
        method: str | list[str],
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Aggregate across rows of the given DataFrame to create a new DataFrame or Series.

        Applies aggregation function(s) given by 'method' across rows (i.e., axis = 0). 

        Args:
            df (pd.DataFrame): The DataFrame.
            method (str | list[str]): The aggregation method. Supported choices: 'min', 'max', 
                'sum', 'mean', 'median', 'count', 'std', 'var', 'prod'.
            cols (list[str] | str | None, optional): Column(s) to aggregate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns to aggregate. Defaults to None.
            suffix (str | None, optional): The suffix of columns to aggregate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns 
                to aggregate. Defaults to None.

        Raises:
            ValueError: If a given aggregation method is unrecognized.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.

        Returns:
            pd.DataFrame | pd.Series: The resulting DataFrame or Series.
        """
        
        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

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

    def _validate_ci_args(
        self,
        method: str,
        alpha: float,
        valid_methods: set[str],
    ) -> tuple[str, float]:
        """Validate the arguments given to compute_ci().

        Additionally converts the string argument to 'method' to lowercase before validating.

        Args:
            method (str): The desired CI-calculation method.
            alpha (float): The desired alpha.
            valid_methods (set[str]): A set of valid methods.

        Raises:
            ValueError: If string argument for 'method' isn't recognized.
            ValueError: If float argument for 'alpha' isn't between 0 and 1 exclusive.

        Returns:
            tuple[str, float]: A tuple 'method' (lowercase) and 'alpha'.
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
        self,
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
            pd.DataFrame: A DataFrame with columns matching those in 'cols' 
                and indices 'point_estimate', 'lower', 'upper', 'count'.
        """
        
        result_matrix = np.vstack([
            point_estimates,
            lower_bounds,
            upper_bounds,
            counts,
        ]) 

        result = pd.DataFrame(
            data = result_matrix,
            columns = cols,
            index = ['point_estimate', 'lower', 'upper', 'count'],
            dtype = float,
        )

        return result    

    def _calc_ci_parametric(
        self,
        df: pd.DataFrame,
        cols: list[str],
        alpha: float,
        distribution: str,
    ) -> pd.DataFrame:
        """Calculate parametric confidence intervals.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str]): Columns on which to operate.
            alpha (float): The desired alpha.
            distribution (str): The distribution to use. Supported choices: 'z', 't'.

        Returns:
            pd.DataFrame: A DataFrame with columns matching those specified in 'cols' and indices 
                'point_estimate', 'lower', 'upper', 'count'.
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


        return self._create_ci_frame(
            cols, 
            stats.loc['mean'].values, # type: ignore
            lower, # type: ignore
            upper, # type: ignore
            stats.loc['count'].values, # type: ignore
        )

    def _calc_ci_proportion(
        self,
        df: pd.DataFrame,
        cols: list[str],
        alpha: float,
        method: str,
    ) -> pd.DataFrame:
        """Calculate proportion confidence intervals.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str]): Columns on which to operate.
            alpha (float): The desired alpha.
            method (str): The CI-calculation method. Supported choices: 'wald', 
                'wilson', 'agresti_coull', 'clopper_pearson' (or 'beta'), 'jeffreys'.

        Returns:
            pd.DataFrame: A DataFrame with columns matching those specified in 'cols' and 
                indices 'point_estimate', 'lower', 'upper', 'count'.
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
            count = stats.loc['sum'].values,
            nobs = stats.loc['count'].values,
            alpha = alpha,
            method = sm_mapping[method],    
        )

        return self._create_ci_frame(
            cols,
            stats.loc['mean'].values, # type: ignore
            lower, # type: ignore
            upper, # type: ignore
            stats.loc['count'].values, # type: ignore
        )
    
    def _calc_ci_bootstrap(
        self,
        df: pd.DataFrame,
        cols: list[str],
        alpha: float,
        method: str,
        metric: str = 'mean',
    ) -> pd.DataFrame:
        """Calculate bootstrap confidence intervals.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str]): Columns on which to operate.
            alpha (float): The desired alpha.
            method (str): The CI-calculation method. Supported choices: 'bootstrap_bca', 
                'bootstrap_percentile', 'bootstrap_basic'.
            metric (str): The measure of central tendency to craft the interval around.
                Supported choices: 'mean', 'median'. Defaults to 'mean'.

        Returns:
            pd.DataFrame: A DataFrame with columns matching those specified in 'cols' and 
                indices 'point_estimate', 'lower', 'upper', 'count'.
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

        stats = df[cols].agg([f'{metric}', 'count', 'sum'], axis = 0)
        metric_func = np.mean
        if metric == 'median':
            metric_func = np.median
        
        for col in cols:
            data = df[col].dropna().values
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
                rng = 0,
            )

            lowers.append(result.confidence_interval.low)
            uppers.append(result.confidence_interval.high)

        return self._create_ci_frame(
            cols,
            stats.loc['mean'].values, # type: ignore
            np.array(lowers), # type: ignore
            np.array(uppers), # type: ignore
            stats.loc['count'].values, # type: ignore
        )

    def calc_ci(
        self,
        df: pd.DataFrame,
        method: str,
        alpha: float = 0.05,
        cols: list[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: str | re.Pattern | None = None,       
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
            cols (list[str] | str | None, optional): Column(s) on which to operate.
                If None, includes all columns. Defaults to None.
            prefix (str | None, optional): The prefix of columns on which to operate. Defaults to None.
            suffix (str | None, optional): The suffix of columns on which to operate. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns on which
                to operate. Defaults to None.

        Raises:
            NotImplementedError: If the CI-calcualtion method is not yet implemented.

        Note:
            Selection parameters (e.g., 'cols', 'prefix', etc.) are used in conjunction with one another, 
            taking the intersection of matching columns. In other words, only columns matching all selection
            criteria will be selected.


        Returns:
            pd.DataFrame: A DataFrame with columns matching those specified in the column-selection parameters 
                and indices 'point_estimate', 'lower', 'upper', 'count'.
        """
        
        df, cols = self._prep_args(df, cols, prefix, suffix, pattern)

        parametric_methods = {'z', 't'}
        proportion_methods = {'wald', 'wilson', 'agresti_coull', 'clopper_pearson', 'beta', 'jeffreys'}
        bootstrap_methods = {'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'}

        method, alpha = self._validate_ci_args(
            method,
            alpha, 
            parametric_methods | proportion_methods | bootstrap_methods
        )

        if method in parametric_methods:
            result = self._calc_ci_parametric(df, cols, alpha, method)

        elif method in proportion_methods:
            result = self._calc_ci_proportion(df, cols, alpha, method)

        elif method in bootstrap_methods:
            result = self._calc_ci_bootstrap(df, cols, alpha, method)

        else:
            raise ValueError(f'CI-calcualtion method \'{method}\' is not recognized.')

        return result  

    # def _apply_p_correction(
    #     self,
    #     df: pd.DataFrame,
    #     correction: str,
    # ):
        
    #     valid_corrections = {'bonferroni', 'holm-bonferroni', 'benjamini-hochberg'}

    #     df, _ = self._prep_args(df, None, None, None, None)

    #     # TODO: add a _validate_p_correction_args() method to handle some of the below?

    #     correction = correction.lower()

    #     if correction not in valid_corrections:
    #         raise ValueError(f'Unrecognized argument for correction: \'{correction}\'. Supported choices: {valid_corrections}.')
        
    #     if 'p_value' not in df.columns:
    #         raise ValueError(f'DataFrame to _apply_correction must have a column \'p_value\' containing p values.')
        
    #     if correction == 'bonferroni':
    #         df['p_value'] *= len(df['p_value'])
    #         # TODO: Test this correction method.

    #     elif correction == 'holm-bonferroni':
    #         raise NotImplementedError(f'Correction method \'{correction}\' is not yet implemented.')
    #         # TODO: Implement correction method

    #     elif correction == 'benjamini-hochberg':
    #         raise NotImplementedError(f'Correction method \'{correction}\' is not yet implemented.')
    #         # TODO: Implement correction method

    #     return df  

    # TODO: Start a new file/class with methods relating to handling graph labels (e.g., wrapping, trimming, etc.)
    # TODO: Start a new file/class with methods relating to creating the graphs (include sorting, pinning, and adding "missing" columns - leaning on categorical dtype?)