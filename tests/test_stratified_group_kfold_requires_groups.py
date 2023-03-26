import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from StratifiedGroupKFoldRequiresGroups import (
    StratifiedGroupKFoldRequiresGroups,
)
import pytest


@pytest.fixture
def data():
    X = np.random.randn(9, 5)
    y = np.array(["class1", "class2", "class3"] * 3)
    groups = np.array(
        [
            "group1",
            "group2",
            "group3",
            "group4",
            "group5",
            "group6",
            "group7",
            "group7",
            "group7",
        ]
    )
    return X, y, groups


def test_functionality(data):
    X, y, groups = data
    internal_cv = StratifiedGroupKFoldRequiresGroups(n_splits=3)
    assert isinstance(internal_cv, StratifiedGroupKFold), "should be subclass"
    assert len(list(internal_cv.split(X, y, groups))) == 3


def test_expected_error(data):
    X, y, groups = data
    internal_cv = StratifiedGroupKFoldRequiresGroups(n_splits=3)
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'groups'"
    ):
        # don't supply arg
        internal_cv.split(X, y)
    with pytest.raises(ValueError, match="requires groups argument"):
        # supply None arg
        internal_cv.split(X, y, groups=None)
