from typing import Optional
from .base import base_model


def train_randomforest_classifier(
        features: str = "features",
        source_table: str = "training",
        target: str = "label",
        option: Optional[str] = None,
        bias: Optional[bool] = None) -> str:
    """Build train_randomforest_classifier query

    Parameters
    -----------

    features :  :obj:`str`
        Feature column name. Default: "features"
    target : :obj:`str`
        Target column for prediction
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`
        An option string for specific algorithm.
    bias : bool
        Add bias or not.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return base_model("train_randomforest_classifier",
                      "feature, weight",
                      features,
                      target,
                      source_table,
                      option,
                      bias)


def train_randomforest_regressor(
        features: str = "features",
        source_table: str = "training",
        target: str = "label",
        option: Optional[str] = None,
        bias: Optional[bool] = None) -> str:
    """Build train_randomforest_classifier query

    Parameters
    -----------

    features :  :obj:`str`
        Feature column name. Default: "features"
    target : :obj:`str`
        Target column for prediction
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`
        An option string for specific algorithm.
    bias : bool
        Add bias or not.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return base_model("train_randomforest_regression",
                      "feature, weight",
                      features,
                      target,
                      source_table,
                      option,
                      bias)
