import os
import pandas as pd
import matplotlib.pyplot as plt
from exceptions import DonneeInvalideError
import csv
import json
import math
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime


class Dataset:
    def __init__(self, path) -> None:
        """Vérifie que les données sont disponibles et valides.
        `path` est un string représentant le chemin du fichier de données (.csv)."""
        if not os.path.isfile(path):
            raise FileNotFoundError("Le fichier de données est introuvable.")
        if os.path.splitext(path)[1].lower() != ".csv":
            raise ValueError("Le fichier de données doit être un fichier CSV.")
        self._path = path
        self._df = self.load_raw_data()
        Dataset.check_data_sanity(self._df)
        print("#### Dataset chargé. Aucune valeur aberrante. ####")

    @property
    def dataframe(self) -> pd.DataFrame:
        """Représente une copie du dataset en format DataFrame."""
        return self._df.copy()

    def load_raw_data(self) -> pd.DataFrame:
        """Vérifie que les données sont disponibles et les retourne en DataFrame.
        Elle renvoie un FileNotFoundError si le fichier de données n'existe pas."""
        return pd.read_csv(self._path, encoding="cp1252")

    @staticmethod
    def check_data_sanity(df: pd.DataFrame) -> None:
        """Vérifie qu'il n'y a aucune donnée dans `df` manquante, ni de données ayant une valeur incorrect.
        Retourne une DonneeInvalideError si au moins une donnée est invalide"""
        if (df.isnull().sum() > 0).any():
            raise DonneeInvalideError("Valeurs manquantes dans le set de données.")
        if (df["Hour"] < 0).any() or (df["Hour"] > 23).any():
            raise DonneeInvalideError("Valeurs de l'heure hors de 0-23h.")
        if (df["Humidity(%)"] < 0).any() or (df["Humidity(%)"] > 100).any():
            raise DonneeInvalideError("Valeurs de l'humidité hors de 0-100%.")
        if (df["Rented Bike Count"] < 0).any():
            raise DonneeInvalideError("Valeurs du nombre de vélo loués sous 0.")
        if (df["Wind speed (m/s)"] < 0).any():
            raise DonneeInvalideError("Valeurs de la vitesse du vent sous 0.")
        if (df["Solar Radiation (MJ/m2)"] < 0).any():
            raise DonneeInvalideError("Valeurs des radiations solaires sous 0.")

    def plot_rented_bikes_per_hour(self) -> None:
        """Crée un graphique du nombre de vélos loués par heures."""
        # faire la moyenne des vélos loués par heure
        locations_par_heure = self._df.groupby("Hour")["Rented Bike Count"].mean()
        plt.plot(locations_par_heure.index, locations_par_heure.values)
        plt.xlabel("Heure de la journée")
        plt.ylabel("Nombre moyen de vélos loués")
        plt.title("Locations moyennes de vélos selon l'heure")
        plt.xticks(range(0, 24)) # 24 heures
        plt.grid(True)
        plt.savefig("velo_loues_par_heure.png")
        plt.show()

    def __str__(self):
        return f"Dataset : {len(self._df)} lignes, {len(self._df.columns)} colonnes"


# ----- Validation des entrées de l'utilisateur -----

class Variable(ABC):
    # Classe parente : toutes les variables ont un nom,
    # mais chaque type de variable sera validé différemment.

    def __init__(self, nom: str) -> None:
        self.nom = nom

    @abstractmethod
    def valider(self, texte: str):
        """Retourne une valeur valide ou déclenche une erreur."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.nom


class VariableNumerique(Variable):
    # Pour les champs numériques : heure, humidité, température, etc.

    def __init__(
        self,
        nom: str,
        minimum: float,
        maximum: float,
        entier: bool
    ) -> None:
        super().__init__(nom)

        self.minimum = minimum
        self.maximum = maximum
        self.entier = entier

    def valider(self, texte: str):
        """Convertit le texte en nombre et vérifie ses limites."""

        # Empêche de laisser un champ vide.
        if not texte.strip():
            raise DonneeInvalideError(
                f"{self.nom} : champ vide."
            )

        try:
            # Convertit ce que l'utilisateur écrit en nombre.
            # Accepte 2,5 autant que 2.5.
            valeur = float(
                texte.strip().replace(",", ".")
            )

        except ValueError:
            raise DonneeInvalideError(
                f"{self.nom} : entre un nombre."
            )

        # Empêche d'entrer des valeurs comme NaN ou l'infini.
        if not math.isfinite(valeur):
            raise DonneeInvalideError(
                f"{self.nom} : nombre invalide."
            )

        # Vérifie que la valeur reste dans les limites permises.
        if valeur < self.minimum or valeur > self.maximum:
            raise DonneeInvalideError(
                f"{self.nom} : valeur entre "
                f"{self.minimum:g} et {self.maximum:g}."
            )

        # Exemple : l'heure doit être 17, et non 17.5.
        if self.entier:
            if not valeur.is_integer():
                raise DonneeInvalideError(
                    f"{self.nom} : entier requis."
                )

            return int(valeur)

        return valeur

class VariableCategorielle(Variable):
    # Pour les catégories comme Summer, Winter ou No Holiday.
    def __init__(
        self,
        nom: str,
        valeurs: list[str]
    ) -> None:
        super().__init__(nom)
        self.valeurs = valeurs

    def valider(self, texte: str) -> str:
        """Vérifie que la catégorie existe dans le jeu de données."""

        if texte not in self.valeurs:
            raise DonneeInvalideError(
                f"{self.nom} : choix invalide."
            )

        return texte

    def enregistrer_metadonnees(
        df: pd.DataFrame,
        colonnes: list[str],
        r2: float
    ) -> None:
        """Sauvegarde les règles des champs et les données de l'histogramme."""

        variables = []

        # Examine chacune des 7 variables utilisées par le modèle.
        for nom in colonnes:
            serie = df[nom]

            if pd.api.types.is_numeric_dtype(serie):
                # Pour un nombre, on garde ses limites minimum et maximum.
                information = {
                    "nom": nom,
                    "type": "numerique",
                    "minimum": float(serie.min()),
                    "maximum": float(serie.max()),
                    "entier": bool(
                        pd.api.types.is_integer_dtype(serie)
                    )
                }

            else:
                # Pour une catégorie, on garde les choix possibles.
                # Exemple : Winter, Spring, Summer, Autumn.
                information = {
                    "nom": nom,
                    "type": "categorielle",
                    "valeurs": sorted(
                        serie.dropna().unique().tolist()
                    )
                }

            variables.append(information)

        # Compte le nombre d'heures observées dans chaque
        # intervalle du nombre de vélos loués.
        effectifs, bornes = np.histogram(
            df["Rented Bike Count"],
            bins=30
        )

        # Rassemble les informations utiles à l'interface.
        informations = {
            "variables": variables,
            "r2_test": float(r2),
            "histogramme": {
                "effectifs": effectifs.tolist(),
                "bornes": bornes.tolist()
            }
        }

        # Retrouve le dossier principal du projet.
        racine = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        dossier = os.path.join(racine, "modeles")

        # Crée le dossier au besoin.
        os.makedirs(dossier, exist_ok=True)

        # Sauvegarde les informations pour les prochains lancements.
        with open(
            os.path.join(dossier, "metadonnees.json"),
            "w",
            encoding="utf-8"
        ) as fichier:

            json.dump(
                informations,
                fichier,
                ensure_ascii=False,
                indent=2
            )
