import pandas as pd
import re
from . import cluster
from . import plot
from . import prep
from . import stats
from . import reduction
from . import test
from .selector import Selector
    
class DataProcessor:
    """Create a DataProcessor object to process data.
    """

    def select(
        self,
        labels: list[str] | set[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
    ) -> Selector:
        """Create a Selector instance.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.

        Note:
            If all selection arguments are None, all columns will be selected.
        """

        return Selector(
            labels = labels,
            prefix = prefix,
            suffix = suffix,
            pattern = pattern
        )

    # Constants from prep.py
    PATTERN_ALIDA_OTHER_OE = prep.PATTERN_ALIDA_OTHER_OE

    # Functions from prep.py
    clean_arg = staticmethod(prep.clean_arg)
    clean_df = staticmethod(prep.clean_df)
    rename_cols = staticmethod(prep.rename_cols)
    remove_cols = staticmethod(prep.remove_cols)
    recode_vals = staticmethod(prep.recode_vals)
    remove_verbal_anchors = staticmethod(prep.remove_verbal_anchors)
    bin = staticmethod(prep.bin)
    filter_by_bounds = staticmethod(prep.filter_by_bounds)
    filter_by_iqr = staticmethod(prep.filter_by_iqr)
    filter_straightliners = staticmethod(prep.filter_straightliners)
    
    # Functions from stats.py
    agg_cols = staticmethod(stats.agg_cols)
    agg_rows = staticmethod(stats.agg_rows)
    calc_ci = staticmethod(stats.calc_ci)

    # Functions from test.py
    test_one_sample = staticmethod(test.test_one_sample)
    test_one_sample_proportion = staticmethod(test.test_one_sample_proportion)