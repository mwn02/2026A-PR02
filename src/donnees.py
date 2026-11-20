import os
import pandas as pd
import matplotlib.pyplot as plt
from exceptions import DonneeInvalideError

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