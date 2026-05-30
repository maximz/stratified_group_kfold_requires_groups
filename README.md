# StratifiedGroupKFoldRequiresGroups

[![](https://img.shields.io/pypi/v/StratifiedGroupKFoldRequiresGroups.svg)](https://pypi.python.org/pypi/StratifiedGroupKFoldRequiresGroups)
[![CI](https://github.com/maximz/StratifiedGroupKFoldRequiresGroups/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/maximz/StratifiedGroupKFoldRequiresGroups/actions/workflows/ci.yaml)
[![](https://img.shields.io/github/stars/maximz/StratifiedGroupKFoldRequiresGroups?style=social)](https://github.com/maximz/StratifiedGroupKFoldRequiresGroups)

A small wrapper around scikit-learn's `StratifiedGroupKFold` that makes the
`groups` argument mandatory when calling `split()`.

## What It Is

`StratifiedGroupKFoldRequiresGroups` is a subclass of
`sklearn.model_selection.StratifiedGroupKFold`. It keeps the underlying
cross-validation behavior from scikit-learn, but adds a guardrail: callers must
provide a non-`None` `groups` argument to `split()`.

This is useful when grouped splitting is part of the correctness of a model
evaluation workflow. If a pipeline, estimator, or helper forgets to pass
`groups`, the split should fail immediately instead of silently behaving like a
regular stratified split without group isolation.

## Installation

```bash
pip install StratifiedGroupKFoldRequiresGroups
```

The package requires Python 3.8 or newer and depends on `scikit-learn`.

## Usage

```python
import numpy as np
from StratifiedGroupKFoldRequiresGroups import StratifiedGroupKFoldRequiresGroups

X = np.random.randn(9, 5)
y = np.array(["class1", "class2", "class3"] * 3)
groups = np.array([
    "group1",
    "group2",
    "group3",
    "group4",
    "group5",
    "group6",
    "group7",
    "group7",
    "group7",
])

cv = StratifiedGroupKFoldRequiresGroups(n_splits=3)

for train_index, test_index in cv.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

The constructor is inherited from scikit-learn's `StratifiedGroupKFold`, so use
the same options such as `n_splits`, `shuffle`, and `random_state`.

## Important Behavior

- `cv.split(X, y, groups)` delegates to
  `sklearn.model_selection.StratifiedGroupKFold.split()`.
- `cv.split(X, y)` raises a `TypeError` because `groups` is a required
  positional argument.
- `cv.split(X, y, groups=None)` raises a `ValueError`.
- The wrapper does not change scikit-learn's splitting algorithm; it only
  enforces that group labels are supplied.

## Development

```bash
pip install -r requirements_dev.txt
pip install -e .
pytest
```

The test suite checks that the class is a `StratifiedGroupKFold` subclass, that
it produces the expected number of splits, and that missing or `None` groups
fail as intended.
