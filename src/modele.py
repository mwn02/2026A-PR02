from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline


class Model(ABC):
    """Prépare et sépare les données communes aux modèles."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.modele: Pipeline | None = None
        self.preparer_donnees(df)

    def preparer_donnees(self, df: pd.DataFrame) -> None:
        """Sépare les données d'entraînement x, y."""
        self.x = df[[
            "Hour", "Humidity(%)", "Wind speed (m/s)", "Temperature(°C)",
            "Seasons", "Holiday", "Rainfall(mm)",
        ]].copy()
        self.y = df["Rented Bike Count"].copy()
        self.separer_donnees()

    def separer_donnees(self) -> None:
        """Divise les données en 89% pour l'entraînement et 20% pour les tests, et reset le modèle entraîné."""
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.20, random_state=42
        )
        self.modele = None

    @staticmethod
    def creer_pretraitement() -> ColumnTransformer:
        """Retourne les données x_train ajustée pour l'entraînement."""
        return ColumnTransformer([
            ("numeriques", StandardScaler(), [
                "Hour", "Humidity(%)", "Wind speed (m/s)",
                "Temperature(°C)", "Rainfall(mm)",
            ]),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["Seasons", "Holiday"]),
        ])

    @abstractmethod
    def entrainer(self) -> Pipeline:
        """Entraîne et conserve le modèle, puis le retourne."""
        raise NotImplementedError

    def predict(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Retourne un tableau 1D de prédictions, une par ligne de x.

        Lève NotFittedError si le modèle n'a pas encore été entraîné.
        """
        try:
            return self.modele.predict(x)
        except(AttributeError):
            raise NotFittedError("Le modèle doit être entraîné avant de prédire.")


class DummyModel(Model):
    def entrainer(self) -> Pipeline:
        """Retourne et conserve le modèle entraîné qui prédit la moyenne de y_train."""
        modele = make_pipeline(self.creer_pretraitement(), DummyRegressor(strategy="mean"))
        modele.fit(self.x_train, self.y_train)
        self.modele = modele
        return self.modele


class SGDModel(Model):
    def entrainer(self) -> Pipeline:
        """Retourne et conserve le modèle SGD entraîné,"""
        modele = make_pipeline(self.creer_pretraitement(), SGDRegressor(random_state=42))
        modele.fit(self.x_train, self.y_train)
        self.modele = modele
        return self.modele
