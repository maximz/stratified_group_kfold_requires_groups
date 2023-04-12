from __future__ import annotations

"""StratifiedGroupKFoldRequiresGroups."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.1"

from sklearn.model_selection import StratifiedGroupKFold
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    # skip unnecessary import unless running type checker
    import numpy as np


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
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Iterator[Any]:
        """Calls StratifiedGroupKFold.split() after verifying that a groups argument was provided."""
        if groups is None:
            raise ValueError(
                "StratifiedGroupKFoldRequiresGroups requires groups argument to be provided to split()."
            )
        return super().split(X, y, groups)
