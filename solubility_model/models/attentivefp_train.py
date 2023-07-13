"""Functions to train AttentiveFP models"""

import os
import glob
import pandas as pd
import numpy as np
import deepchem as dc
from deepchem.models import AttentiveFPModel
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from solubility_model.utils.paths import make_dir_function as mdf


class EarlyStopper:
    """Tracks validation loss and decides when to stop training"""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Stops the model training if no improvement in validation loss

        Args:
            validation_loss (float): Validation loss from model training

        Returns:
            bool: Whether to continue training
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def regression_model(num_epochs=800, patience=30, min_delta=0, cv=5):
    """Fit an AttentiveFP regression model to the dataset

    Args:
        num_epochs (int, optional): Max number of epochs to train for. Defaults to 800.
        patience (int, optional): Number of epochs with no improvement before early stopping. Defaults to 30.
        min_delta (int, optional): . Defaults to 0.
        cv (int, optional): Number of folds in cross validation. Defaults to 5.
    """
    # Load data
    training_data = pd.read_csv(mdf("data/processed/train_smiles.csv")())

    # Generate scaffolds
    mols = [Chem.MolFromSmiles(x) for x in training_data.smiles.to_list()]
    scaffolds = [Chem.MolToSmiles(GetScaffoldForMol(mol)) for mol in mols]

    # Metrics
    R2 = []
    RMSE = []
    MSE = []
    MAE = []

    # Featurize data
    print("Featurizing data")
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    X = featurizer.featurize(training_data.smiles.to_list())
    training_dataset = dc.data.NumpyDataset(X=X, y=training_data.y_true)

    # Featurize test set
    test_data = pd.read_csv(mdf("data/processed/test_smiles.csv")())
    test_smiles = test_data.smiles
    y_test = test_data.y_true
    X_test = featurizer.featurize(test_smiles)
    test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)

    # Train-valid split
    gss = GroupShuffleSplit(n_splits=cv, test_size=0.2, random_state=0)
    for fold in range(cv):
        print(f"Fitting fold: {fold}")
        train_index, valid_index = list(
            gss.split(training_data.smiles, groups=scaffolds)
        )[fold]
        valid_data = training_data.iloc[valid_index].reset_index(drop=True)
        train_dataset = training_dataset.select(train_index)
        valid_dataset = training_dataset.select(valid_index)
        assert (
            len(
                ({scaffolds[i] for i in set(train_index)}).intersection(
                    ({scaffolds[i] for i in set(valid_index)})
                )
            )
            == 0
        )

        # Clear tmp folder
        files = glob.glob(f"{mdf('models/tmp')()}/*")
        for f in files:
            os.remove(f)

        # Training model
        model = AttentiveFPModel(
            mode="regression", n_tasks=1, batch_size=16, learning_rate=0.001
        )

        train_losses = []
        valid_losses = []
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

        for epoch in range(num_epochs):
            train_loss = model.fit(train_dataset, nb_epoch=1)
            valid_pred = model.predict(valid_dataset)
            valid_loss = mean_squared_error(
                valid_data.y_true, valid_pred, squared=False
            )
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            model.save_checkpoint(
                max_checkpoints_to_keep=patience + 1,
                model_dir=f"{(mdf('models/tmp/')())}/",
            )
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Training loss = {train_loss:.3f}, validation RMSE = {valid_loss:.3f},"
                )
            if early_stopper.early_stop(valid_loss):
                best_epoch = valid_losses.index(min(valid_losses))
                print(
                    f"Early stopping at epoch {epoch}. Best epoch {best_epoch}, validation loss: {min(valid_losses)}"
                )
                break

        # Make predictions
        best_model_checkpoint = len(valid_losses) - valid_losses.index(
            min(valid_losses)
        )

        model.restore(f"{(mdf('models/tmp/')())}/checkpoint{best_model_checkpoint}.pt")
        model.save_checkpoint(model_dir=f"{mdf('models/AttentiveFP')()}/fold_{fold}/")
        y_pred = model.predict(test_dataset)

        # Evaluate performance on test set
        y_true = y_test
        R2.append(r2_score(y_true, y_pred))
        RMSE.append(mean_squared_error(y_true, y_pred, squared=False))
        MSE.append(mean_squared_error(y_true, y_pred))
        MAE.append(mean_absolute_error(y_true, y_pred))
        metrics = {
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
            "MSE": float(mean_squared_error(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
        }
        print(f"Fold {fold}: {metrics}")

    print("Average model performance:")
    print(f"R2: {np.mean(R2):.3f}+/-{np.std(R2):.4}")
    print(f"RMSE: {np.mean(RMSE):.3f}+/-{np.std(RMSE):.4}")
    print(f"MSE: {np.mean(MSE):.3f}+/-{np.std(MSE):.4}")
    print(f"MAE: {np.mean(MAE):.3f}+/-{np.std(MAE):.4}")


def make_predictions(cv=5):
    """Uses an AttentiveFP regression model to make predictions

    Args:
        cv (int, optional): Number of folds in cross validation. Defaults to 5.
    """

    # Metrics
    R2 = []
    RMSE = []
    MSE = []
    MAE = []

    # Featurize test set
    print("Featurizing test set")
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    test_data = pd.read_csv(mdf("data/processed/test_smiles.csv")())
    test_smiles = test_data.smiles
    y_test = test_data.y_true
    X_test = featurizer.featurize(test_smiles)
    test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)

    for fold in range(cv):
        print(f"Predicting fold: {fold}")

        # Prepare model
        model = AttentiveFPModel(
            mode="regression", n_tasks=1, batch_size=16, learning_rate=0.001
        )

        # Make predictions
        model.restore(f"{mdf('models/AttentiveFP')()}/fold_{fold}/checkpoint1.pt")
        y_pred = model.predict(test_dataset)

        # Evaluate performance on test set
        y_true = y_test
        R2.append(r2_score(y_true, y_pred))
        RMSE.append(mean_squared_error(y_true, y_pred, squared=False))
        MSE.append(mean_squared_error(y_true, y_pred))
        MAE.append(mean_absolute_error(y_true, y_pred))
        metrics = {
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
            "MSE": float(mean_squared_error(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
        }
        print(f"Fold {fold}: {metrics}")

    print("Average model performance:")
    print(f"R2: {np.mean(R2):.3f}+/-{np.std(R2):.4}")
    print(f"RMSE: {np.mean(RMSE):.3f}+/-{np.std(RMSE):.4}")
    print(f"MSE: {np.mean(MSE):.3f}+/-{np.std(MSE):.4}")
    print(f"MAE: {np.mean(MAE):.3f}+/-{np.std(MAE):.4}")
