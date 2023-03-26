"""StratifiedGroupKFoldRequiresGroups."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.1"

from sklearn.model_selection import StratifiedGroupKFold


class StratifiedGroupKFoldRequiresGroups(StratifiedGroupKFold):
    """
    Wrapper around sklearn.model_selection.StratifiedGroupKFold that requires groups argument to be provided to split().
    Otherwise we are not getting the stated benefits of splitting by group.
    This ensures that we are using the CV splitter in the intended way, and are never passing in a None value for groups.

    This helps avoid bugs like instantiating a model with a StratifiedGroupKFold CV splitter inside,
    but then during a later fit() operation, not having the model call split() with a groups parameter.
    We might not detect the error otherwise.
    """

    def split(
        self,
        X,
        y,
        groups,
    ):
        """Calls StratifiedGroupKFold.split() after verifying that a groups argument was provided."""
        if groups is None:
            raise ValueError(
                "StratifiedGroupKFoldRequiresGroups requires groups argument to be provided to split()."
            )
        return super().split(X, y, groups)
