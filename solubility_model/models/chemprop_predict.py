"""Functions for making predictions"""

import chemprop
from solubility_model.utils.paths import make_dir_function as mdf


def basic_model_predict():
    """Uses the basic chemprop model to make predictions on unseen test data.
    Saves the predictions in models/predictions/
    """
    arguments = [
        "--test_path",
        str(mdf("data/processed/test_smiles.csv")()),
        "--preds_path",
        str(mdf("models/predictions/test_predictions_no_features_model.csv")()),
        "--checkpoint_dir",
        str(mdf("models/no_features_model")()),
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    _ = chemprop.train.make_predictions(args=args)


def rdkit2d_model_predict():
    """Uses the basic chemprop model to make predictions on unseen test data.
    Saves the predictions in models/predictions/
    """
    arguments = [
        "--test_path",
        str(mdf("data/processed/test_smiles.csv")()),
        "--preds_path",
        str(mdf("models/predictions/test_predictions_rdkit2d_model.csv")()),
        "--checkpoint_dir",
        str(mdf("models/rdkit2d_model")()),
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    _ = chemprop.train.make_predictions(args=args)


def morg_model_predict():
    """Uses the basic chemprop model to make predictions on unseen test data.
    Saves the predictions in models/predictions/
    """
    arguments = [
        "--test_path",
        str(mdf("data/processed/test_smiles.csv")()),
        "--preds_path",
        str(mdf("models/predictions/test_predictions_morg_model.csv")()),
        "--checkpoint_dir",
        str(mdf("models/morg_model")()),
        "--features_generator",
        "morgan_count",
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    _ = chemprop.train.make_predictions(args=args)
