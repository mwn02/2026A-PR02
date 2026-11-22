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
import json
from donnees import Historique, VariableNumerique, VariableCategorielle
from exceptions import DonneeInvalideError

def timer(func):
    """Mesure le temps d'entrainement et retourne le modele."""
    def wrapper(self):
        debut = perf_counter()
        resultat = func(self)
        print(f"Entraînement de {type(self).__name__} - temps écoulé: {perf_counter() - debut:.4f} secondes.")
        return resultat

    return wrapper


class Model(ABC):
    """Classe abstraite commune aux modèles de régression.

    L'initialisation prépare les données et crée un partage reproductible
    80 % entraînement / 20 % test. Les sous-classes implémentent entrainer().
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.modele: Pipeline | None = None
        self.preparer_donnees(df)

    def preparer_donnees(self, df: pd.DataFrame) -> None:
        """Copie les variables explicatives et la cible, puis les sépare.

        Réinitialise le modèle entraîné. Lève KeyError si une colonne requise manque.
        """
        try:
            self.x = df[[
                "Hour", "Humidity(%)", "Wind speed (m/s)", "Temperature(°C)",
                "Seasons", "Holiday", "Rainfall(mm)",
            ]].copy()
            self.y = df["Rented Bike Count"].copy()
        except(KeyError):
            raise KeyError("Une ou plusieurs colonnes du fichier de données sont manquantes.")
        self.separer_donnees()

    def separer_donnees(self) -> None:
        """Divise les données en 80% pour l'entraînement et 20% pour les tests, et reset le modèle entraîné.
        La reproductibilité des résultats est assurée."""
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.20, random_state=42
        )
        self.modele = None

    def save(self) -> Path:
        """Sauvegarde le modèle entraîné et retourne son chemin.

        Sauvegardé avec modeles/<nom de classe>.joblib.
        Crée les dossiers parents et remplace le fichier s'il existe déjà.
        Lève NotFittedError avant l'entraînement et propage les erreurs d'écriture.
        """
        if self.modele is None:
            raise NotFittedError("Le modèle doit être entraîné avant de sauvegarder.")
        destination = Path("modeles") / f"{type(self).__name__}.joblib"
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.modele, destination)
        return destination

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
    """Modèle qui prédit la moyenne de la cible du jeu d'entraînement."""

    @timer
    def entrainer(self) -> Pipeline:
        """Retourne et conserve le modèle entraîné qui prédit la moyenne de y_train."""
        modele = make_pipeline(self.creer_pretraitement(), DummyRegressor(strategy="mean"))
        modele.fit(self.x_train, self.y_train)
        self.modele = modele
        return self.modele


class SGDModel(Model):
    """Régression linéaire entraînée par descente de gradient stochastique."""

    @timer
    def entrainer(self) -> Pipeline:
        """Retourne et conserve le modèle SGD entraîné,"""
        modele = make_pipeline(self.creer_pretraitement(), SGDRegressor(random_state=42))
        modele.fit(self.x_train, self.y_train)
        self.modele = modele
        return self.modele


class ApplicationVelos:

    def __init__(self) -> None:
        """Charge le modèle, les métadonnées et prépare l'historique."""

        # Retrouve le dossier principal du projet.
        racine = Path(__file__).resolve().parent.parent

        # Indique où se trouvent le modèle et ses informations.
        chemin_modele = racine / "modeles" / "SGDModel.joblib"
        chemin_infos = racine / "modeles" / "metadonnees.json"

        # Vérifie que les deux fichiers existent avant de les charger.
        if not chemin_modele.is_file() or not chemin_infos.is_file():
            raise FileNotFoundError(
                "Modèle ou métadonnées introuvables. Lance d'abord main.py."
            )

        # Recharge le modèle sauvegardé par ton coéquipier.
        # Aucun nouvel entraînement n'est nécessaire.
        self._modele = joblib.load(chemin_modele)

        # Charge les informations des 7 variables depuis le JSON.
        with open(chemin_infos, "r", encoding="utf-8") as fichier:
            self._infos = json.load(fichier)

        # Prépare le fichier CSV qui conservera les prédictions.
        chemin_historique = racine / "historique" / "predictions.csv"
        self._historique = Historique(str(chemin_historique))

        # Liste qui contiendra les variables utilisées par l'interface.
        self.variables = []

        # Crée chaque variable selon son type.
        for info in self._infos["variables"]:

            if info["type"] == "numerique":
                # Exemple : heure, température, humidité.
                variable = VariableNumerique(
                    info["nom"],
                    info["minimum"],
                    info["maximum"],
                    info["entier"]
                )

            else:
                # Exemple : saison et jour férié.
                variable = VariableCategorielle(
                    info["nom"],
                    info["valeurs"]
                )

            self.variables.append(variable)

    def predire(self, saisies: dict[str, str]) -> int:
        """Valide les saisies, calcule et enregistre une prédiction."""

        valeurs = {}

        # Vérifie chaque valeur entrée par l'utilisateur.
        for variable in self.variables:
            texte = saisies.get(variable.nom, "")
            valeurs[variable.nom] = variable.valider(texte)

        # Transforme les valeurs en DataFrame.
        # Le modèle scikit-learn attend ce format.
        tableau = pd.DataFrame([valeurs])

        # Utilise le modèle entraîné pour prédire le nombre de vélos.
        prediction = float(self._modele.predict(tableau)[0])

        # Vérifie que le résultat est un nombre valide.
        if not np.isfinite(prediction):
            raise DonneeInvalideError(
                "Le modèle a retourné un nombre invalide."
            )

        # Le nombre de vélos ne peut pas être négatif.
        # On ramène les prédictions négatives à 0 et on arrondit.
        resultat = round(max(0, prediction))

        # Enregistre les valeurs et le résultat dans l'historique.
        self._historique.ajouter(valeurs, resultat)

        # Retourne le nombre de vélos pour l'afficher dans l'interface.
        return resultat

    def lire_historique(self) -> list[dict]:
        """Récupère toutes les anciennes prédictions."""

        return self._historique.lire()

    def donnees_histogramme(self) -> dict:
        """Retourne les données nécessaires au graphique."""

        return self._infos["histogramme"]