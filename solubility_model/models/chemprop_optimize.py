"""Functions for optimizing a chemprop model"""

import chemprop
from solubility_model.utils.paths import make_dir_function as mdf


def optimize_model():
    """Optimizes a chemprop model to predict solubility using in-house vernalis data.
    Includes RDKit2D features.
    Saves the model in models/optimized_model/
    """
    import chemprop
    from solubility_model.utils.paths import make_dir_function as mdf

    arguments = [
        "--data_path",
        str(mdf("data/processed/train_smiles.csv")()),
        "--dataset_type",
        "regression",
        "--num_iters",
        "20",
        "--config_save_path",
        str(mdf("models/chemprop_optimized_config")()),
        "--split_type",
        "scaffold_balanced",
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
    ]
    args = chemprop.args.HyperoptArgs().parse_args(arguments)
    chemprop.hyperparameter_optimization.hyperopt(args=args)

    arguments = [
        "--data_path",
        str(mdf("data/processed/train_smiles.csv")()),
        "--dataset_type",
        "regression",
        "--save_dir",
        str(mdf("models/optimized_model")()),
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
        "--config_path",
        str(mdf("models/chemprop_optimized_config")()),
        "--split_type",
        "scaffold_balanced",
        "--num_folds",
        "5",
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    _, _ = chemprop.train.cross_validate(
        args=args, train_func=chemprop.train.run_training
    )


def optimized_model_predict():
    """Uses the optimized chemprop model to make predictions on unseen test data.
    Saves the predictions in models/predictions/
    """
    arguments = [
        "--test_path",
        str(mdf("data/processed/test_smiles.csv")()),
        "--preds_path",
        str(mdf("models/predictions/test_predictions_optimized_model.csv")()),
        "--checkpoint_dir",
        str(mdf("models/optimized_model")()),
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    _ = chemprop.train.make_predictions(args=args)
