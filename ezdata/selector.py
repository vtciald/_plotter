import re
import pandas as pd

class Selector:
    """Create a Selector object to select columns.
    """

    def __init__(
        self,
        *,
        labels: list[str] | set[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
    ) -> None:
        """Initialize a Selector instance.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.

        Note:
            If all selection arguments are None, all columns will be selected.
        """

        self.labels = set(labels) if isinstance(labels, (list, set)) else set([labels]) if labels else None
        self.prefix = prefix
        self.suffix = suffix
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    
    def __call__(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Resolve column selection from Selector.

        Args:
            df (pd.DataFrame): The DataFrame.

        Returns:
            list[str]: The list of selected column labels.
        """
        
        all_cols = df.columns.tolist()
        
        if self.labels is None and self.prefix is None and self.suffix is None and self.pattern is None:
            return all_cols
        
        matched_cols = [
            col for col in all_cols
            if (self.labels is None or col in self.labels)
            and (self.prefix is None or col.startswith(self.prefix))
            and (self.suffix is None or col.endswith(self.suffix))
            and (self.pattern is None or re.search(self.pattern, col))
        ]

        return matched_cols