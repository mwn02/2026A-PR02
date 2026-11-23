
from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter

import joblib
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


def timer(func):
    """Mesure la durée d'exécution de la méthode d'entraînement."""

    # Le décorateur (timer) exécute la méthode reçue et mesure sa durée.
    def wrapper(self):
        # Enregistre le temps de départ avant de lancer l'entraînement.
        debut = perf_counter()
        # exécute la méthode entrainer() du modèle concerné
        resultat = func(self)

        print(f"Entraînement de {type(self).__name__} - temps écoulé: {perf_counter() - debut:.4f} secondes.")
        # retourne le resultat de l'entrainement
        return resultat
    # permet au décorateur de remplacer la méthode par sa version chronométrée.
    return wrapper


class Model(ABC):
    """Classe parente des modèles DummyModel et SGDModel.
    Prépare les données et regroupe les méthodes communes aux deux modèles.
    """

    def __init__(self, df: pd.DataFrame) -> None:

        # Le modèle n'est pas encore entraîné.
        # Prépare les données dès la création de l'objet.
        self.modele: Pipeline | None = None
        self.preparer_donnees(df)

    def preparer_donnees(self, df: pd.DataFrame) -> None:
        """Sélectionne les variables d'entrée X et la cible y."""

        try:
            # X contient les 7 informations utilisées pour prédire.
            self.x = df[[
                "Hour", "Humidity(%)", "Wind speed (m/s)", "Temperature(°C)",
                "Seasons", "Holiday", "Rainfall(mm)",
            ]].copy()

            # y contient le nombre réel de vélos qu'on veut prédire.
            self.y = df["Rented Bike Count"].copy()

        except KeyError:
            raise KeyError("Une ou plusieurs colonnes du fichier de données sont manquantes.")

        # Une fois X et y préparés, les sépare pour l'entraînement et l'évaluation.
        self.separer_donnees()

    def separer_donnees(self) -> None:
        """Réserve 80 % des données à l'entraînement et 20 % aux tests."""

        # random_state=42 permet de retrouver la même séparation à chaque exécution.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.20, random_state=42
        )

        # Réinitialise le modèle puisqu'il n'est pas entraîné sur cette séparation.
        self.modele = None

    def save(self) -> Path:
        """Sauvegarde le modèle entraîné dans le dossier modeles/."""

        if self.modele is None:
            raise NotFittedError("Le modèle doit être entraîné avant de sauvegarder.")

        # Crée un fichier Joblib portant le nom de la classe.
        # Ce fichier permettra de recharger le modèle sans le réentraîner.
        destination = Path("modeles") / f"{type(self).__name__}.joblib"
        destination.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.modele, destination)
        return destination

    @staticmethod
    # Cette méthode prépare les transformations des variables.
    # Elle n'a pas besoin de self, car elle n'utilise aucune information de l'objet.
    def creer_pretraitement() -> ColumnTransformer:
        """Définit comment préparer les variables avant l'entraînement."""

        # StandardScaler ajuste l'échelle des variables numériques.
        # OneHotEncoder transforme les catégories en colonnes numériques.
        # ColumnTransformer associe chaque transformation aux bonnes colonnes.
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
        """Oblige chaque classe enfant (DummyModel et SGDModel) à définir son propre entraînement."""
        raise NotImplementedError

    def predict(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Utilise le modèle entraîné pour prédire les locations de vélos."""

        try:
            # Le pipeline applique le prétraitement puis effectue la prédiction.
            return self.modele.predict(x)

        except AttributeError:
            raise NotFittedError("Le modèle doit être entraîné avant de prédire.")


class DummyModel(Model):
    """Modèle de référence qui prédit toujours la moyenne des locations."""

    @timer
    def entrainer(self) -> Pipeline:

        # Associe le prétraitement au DummyRegressor dans un pipeline.
        modele = make_pipeline(self.creer_pretraitement(), DummyRegressor(strategy="mean"))

        # fit() apprend la moyenne de y_train à partir des données d'entraînement.
        modele.fit(self.x_train, self.y_train)

        # Conserve le modèle entraîné pour predict() et save().
        self.modele = modele
        return self.modele


class SGDModel(Model):
    """Modèle de régression qui apprend à prédire à partir des 7 variables."""

    @timer
    def entrainer(self) -> Pipeline:

        # Associe le prétraitement au SGDRegressor dans un pipeline.
        modele = make_pipeline(self.creer_pretraitement(), SGDRegressor(random_state=42))

        # fit() entraîne le SGD pour apprendre la relation entre X et y.
        modele.fit(self.x_train, self.y_train)

        # Conserve le modèle entraîné pour predict() et save().
        self.modele = modele
        return self.modele
