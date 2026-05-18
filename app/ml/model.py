"""Machine-learning model placeholder.

Replace the ``DummyModel`` class with your actual model implementation.
Common patterns:
  - Load a pre-trained scikit-learn / joblib model from disk.
  - Wrap a PyTorch or TensorFlow model.
  - Call an external inference service.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DummyModel:
    """Placeholder model — returns the mean of the input features.

    Replace this with a real model before going to production.
    """

    name: str = "dummy-v0"

    def predict(self, features: list[float]) -> Any:
        arr = np.array(features, dtype=float)
        return float(arr.mean())


# Module-level singleton — swap this out for your real model loader.
model = DummyModel()
