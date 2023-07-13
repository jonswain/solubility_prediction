"""Functions for training models"""

import chemprop
from solubility_model.utils.paths import make_dir_function as mdf


def chemprop_no_features():
    """Trains a chemprop model to predict solubility using in-house vernalis data.
    Includes no extra features.
    Saves the model in models/no_features_model/
    """
    arguments = [
        "--data_path",
        str(mdf("data/processed/train_smiles.csv")()),
        "--dataset_type",
        "regression",
        "--save_dir",
        str(mdf("models/no_features_model")()),
        "--split_type",
        "scaffold_balanced",
        "--num_folds",
        "5",
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    _, _ = chemprop.train.cross_validate(
        args=args, train_func=chemprop.train.run_training
    )


def chemprop_rdkit2d():
    """Trains a chemprop model to predict solubility using in-house vernalis data.
    Includes chemprop RDKit 2D descriptors.
    Saves the model in models/rdkit_model/
    """
    arguments = [
        "--data_path",
        str(mdf("data/processed/train_smiles.csv")()),
        "--dataset_type",
        "regression",
        "--save_dir",
        str(mdf("models/rdkit2d_model")()),
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
        "--split_type",
        "scaffold_balanced",
        "--num_folds",
        "5",
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    _, _ = chemprop.train.cross_validate(
        args=args, train_func=chemprop.train.run_training
    )


def chemprop_morg():
    """Trains a chemprop model to predict solubility using in-house vernalis data.
    Includes chemprop Morgan count fingerprints.
    Saves the model in models/rdkit_model/
    """
    arguments = [
        "--data_path",
        str(mdf("data/processed/train_smiles.csv")()),
        "--dataset_type",
        "regression",
        "--save_dir",
        str(mdf("models/morg_model")()),
        "--features_generator",
        "morgan_count",
        "--split_type",
        "scaffold_balanced",
        "--num_folds",
        "5",
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    _, _ = chemprop.train.cross_validate(
        args=args, train_func=chemprop.train.run_training
    )
