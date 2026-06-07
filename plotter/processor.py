import pandas as pd
import numpy as np
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
    PATTERN_HOW_BIN = re.compile(r'(?P<kind>i|q)(?P<number>\d+\.?\d*)', re.IGNORECASE)

    def _prep_args(
        self, 
        df: pd.DataFrame, 
        cols: list[str] | str | None = None,
    ) -> tuple[pd.DataFrame, list]:
        """Prepare df and cols arguments.

        Helper method to handle DataFrame copying and column selection.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | str | None): A list of strings column names or a single string column name.

        Returns:
            tuple[pd.DataFrame, list[str]]: A tuple containing a copy of the DataFrame and the list of column names.
        """
        
        df = df.copy()

        if cols is None:
            cols = df.columns.tolist()

        elif isinstance(cols, str):
            cols = [cols]

        return df, cols
    
    def remove_cols(
        self, 
        df: pd.DataFrame, 
        pattern: re.Pattern, 
        cols: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Remove columns whose labels match the given regex pattern.

        Args:
            df (pd.DataFrame): The DataFrame.
            pattern (re.Pattern): A compiled regex pattern.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame with columns removed.
        """
        
        df, cols = self._prep_args(df, cols)
        
        remove_cols = [col for col in cols if re.search(pattern, col)]

        df = df.drop(columns = remove_cols)

        return df
    
    def fix_characters_df(
        self,
        df: pd.DataFrame,
        cols: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Standardize characters and strip strings in DataFrame.

        Applies character translation and stripping to column names and values of object or string columns.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame with standardized characters.
        """

        df, cols = self._prep_args(df, cols)

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
    ) -> pd.DataFrame:
        """Extract leading digits from string values in a DataFrame.

        Takes leading digits from values in string or object columns and coerces the columns to numeric.

        Args:
            df (pd.DataFrame): The DataFrame.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The updated DataFrame.
        """

        df, cols = self._prep_args(df, cols)

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
    ) -> pd.DataFrame:
        """Replace straightliners' values with NaN.

        Args:
            df (pd.DataFrame): The DataFrame.
            min_unique (int, optional): The minimum number of unique values desired in a row. 
                If below this number, values will be replaced with NaN. Defaults to 2.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame with straightliners' values replaced.
        """

        df, cols = self._prep_args(df, cols)

        df[cols] = df[cols].where(df[cols].nunique(axis = 1) >= min_unique)

        return df
    
    def bin(
        self,
        df: pd.DataFrame, 
        how: str | list,
        cols: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Bin DataFrame values.

        Bins on the basis of number of quantiles, number of equal-width intervals, or explicitly given edges.

        Args:
            df (pd.DataFrame): The DataFrame.
            how (str | list): The binning method to apply. Options include:
                - A string of the form 'q#': Quantile binning (e.g., 'q4' and 'q2' bins on the basis of quartiles or a median split, respectively).
                - A string of the form 'i#': Interval binning (e.g., 'i5' will create 5 equal-width intervals that capture the range of values).
                - A list of numbers: Explicitly defined bin edges.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The binned DataFrame.
        """

        df, cols = self._prep_args(df, cols)

        if isinstance(how, str):
            df = self._bin_by_string(df, how, cols = cols)
            
        elif isinstance(how, list):
            df = self._bin_by_edges(df, how , cols = cols)

        return df
    
    def _bin_by_string(
        self,
        df: pd.DataFrame,
        how: str,
        cols: list[str],
    ) -> pd.DataFrame:
        """Bin DataFrame values based on the given string.

        Helper method to handle str input for 'how' arg of bin().

        Args:
            df (pd.DataFrame): The DataFrame.
            how (str): String representing the desired binning method, of the form 'q#' or 'i#'.
            cols (list[str]): A list of columns on which to operate. 

        Returns:
            pd.DataFrame: The binned DataFrame.
        """

        how_match = re.match(self.PATTERN_HOW_BIN, how)

        if not how_match: 
            raise ValueError(f'String argument \'{how}\' for parameter \'how\' doesn\'t follow the expected pattern: \'q#\' or \'i#\'.')
        
        how_kind = how_match.group('kind')
        how_number = float(how_match.group('number'))

        if how_number.is_integer():
            how_number = int(how_number)

        elif isinstance(how_number, float):
            how_number = int(how_number)
            warnings.warn(f'String argument for parameter \'how\' included a float. \'{how}\' was converted to \'{how_kind}{how_number}\'')
        
        for col in cols:
            if how_kind == 'q':
                df[col] = pd.qcut(df[col], how_number).astype(str)

            elif how_kind == 'i':
                df[col] = pd.cut(df[col], how_number).astype(str)

        return df

    def _bin_by_edges(
        self,
        df: pd.DataFrame,
        how: list[int | float],
        cols: list[str],
    ) -> pd.DataFrame:
        """Bin DataFrame values based on the given edges.

        Helper method to handle list input for 'how' arg of bin().

        Args:
            df (pd.DataFrame): The DataFrame.
            how (list[int | float]): List of bin edges.
            cols (list[str]): A list of columns on which to operate. 

        Returns:
            pd.DataFrame: The binned DataFrame.
        """

        for col in cols:
            df[col] = pd.cut(df[col], how).astype(str)

        return df

    
    def filter_by_bounds(
        self,
        df: pd.DataFrame,
        min_val: int | None = None,
        max_val: int | None = None,
        cols: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Filter DataFrame values based on the given minimum and maximum.

        Args:
            df (pd.DataFrame): The DataFrame.
            min (int, optional): The minimum value to keep. Defaults to None.
            max (int, optional): The maximum value to keep. Defaults to None.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """

        df, cols = self._prep_args(df, cols)

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
    ) -> pd.DataFrame:
        """Filter DataFrame values based on the IQR method.

        Defines a maximum value as Q3 + factor * IQR and a minimum value as Q1 - factor * IQR 
        where IQR = Q3 - Q1

        Args:
            df (pd.DataFrame): The DataFrame.
            factor (float | int, optional): The factor by which to multiply the IQR to determine 
                min and max values. Defaults to 1.5.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """

        df, cols = self._prep_args(df, cols)

        qs = df[cols].quantile([0.25, 0.75], axis = 0)
        iqrs = qs.iloc[1] - qs.iloc[0]

        mins = qs.iloc[0] - iqrs * factor
        maxes = qs.iloc[1] + iqrs * factor

        df[cols] = df[cols].where((df[cols] <= maxes) & (df[cols] >= mins))

        return df
    
    def _validate_one_arg_used(
        self,
        **kwargs,  
    ) -> None:
        """Validate that only a single argument is not None.

        Returns:
            None: None. Raises a ValueError if all are None. Raises a warning if multiple arguments were not None.
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
            warnings.warn(f'Only one of the following parameters can be used in {class_name}.{caller_name}: {param_list}. Defaulting to the documented precedence.')

        elif len(arg_list) < 1:
            raise ValueError(f'One of the following parameters must be used in {class_name}.{caller_name}: {param_list}. All had a value of None.')

        return None
    
    def get_cols(
        self,
        df: pd.DataFrame,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
    ) -> list[str]:
        """Get a list of column names from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame.
            prefix (str | None, optional): The prefix of desired column names. Defaults to None.
            suffix (str | None, optional): The suffix of desired column names. Defaults to None.
            pattern (re.Pattern | str | None, optional): A regex pattern describing the desired column names. Defaults to None.

        Returns:
            list[str]: The list of desired column names.
        """
        
        self._validate_one_arg_used(prefix = prefix, suffix = suffix, pattern = pattern)
        
        cols = df.columns.tolist()

        if prefix is not None:
            return [col for col in cols if col.startswith(prefix)]
        
        elif suffix is not None:
            return [col for col in cols if col.endswith(suffix)]
        
        elif pattern is not None:
            if isinstance(pattern, str):
                pattern = re.compile(pattern)

            return [col for col in cols if re.search(pattern, col)]
        
        return cols
    
    def _stringify_mapper(
        self,
        mapper: dict[str | re.Pattern, str],
        regex: bool,
        cols: list[str],
    ) -> dict[str, str]:
        """Standardize mapper argument to a dictionary of strings.

        Args:
            mapper (dict[str | re.Pattern, str]): A dictionary mapping strings and/or regex patterns to strings.
            regex (bool): Whether to treat string keys of mapper as regex patterns.
            cols (list[str]): A list of strings column names on which to operate.

        Returns:
            dict[str, str]: _description_
        """
        
        new_mapper = {}
        
        for k, v in mapper.items():

            if isinstance(k, re.Pattern) or regex == True:
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
        regex: bool = False,
        cols: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Rename DataFrame columns according to the given mapper.

        Args:
            df (pd.DataFrame): The DataFrame.
            mapper (dict[str | re.Pattern, str]): A dictionary mapping existing column names to desired column names.
                Existing column names can take the form of strings and/or compiled regex patterns.
            regex (bool, optional): Whether to treat string keys of mapper as regex patterns. Defaults to False.
            cols (list[str] | str | None, optional): A column name or column names on which to operate. 
                If None, operates on all columns. Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame with renamed columns.
        """
        
        df, cols = self._prep_args(df, cols)
        mapper = self._stringify_mapper(mapper, regex, cols)

        df = df.rename(columns = mapper)
        
        return df