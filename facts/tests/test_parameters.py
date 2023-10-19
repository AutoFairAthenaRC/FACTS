from ..parameters import (
    naive_feature_change_builder
)

def test_naive_feature_change_builder() -> None:
    numeric_features = ["a", "b", "c", "d"]
    categorical_features = ["e", "f", "g", "h"]
    feat_weights = {"a": 10, "b": 10, "g": 3, "h": 3}

    fns = naive_feature_change_builder(numeric_features, categorical_features, feat_weights)

    assert all(f in fns.keys() for f in categorical_features)
    assert all(f in fns.keys() for f in numeric_features)

    assert fns["a"](1, 5) == 40
    assert fns["b"](1, 13) == 120
    assert fns["c"](1, 5) == 4
    assert fns["d"](212, 41341) == 41341 - 212
    assert fns["e"](1, 5) == 1
    assert fns["f"](434343, 434343) == 0
    assert fns["g"](1, 5) == 3
    assert fns["h"](5, 5) == 0
