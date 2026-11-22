import os, csv, json, math
import pandas as pd
import matplotlib.pyplot as plt
from exceptions import DonneeInvalideError
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np

class Dataset:
    def __init__(self, path) -> None:
        """Vérifie que le fichier CSV existe et charge ses données."""
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
        """Retourne une copie du jeu de données."""
        return self._df.copy()

    def load_raw_data(self) -> pd.DataFrame:
        """Charge le fichier CSV dans un DataFrame."""
        return pd.read_csv(self._path, encoding="cp1252")

    @staticmethod
    def check_data_sanity(df: pd.DataFrame) -> None:
        """Vérifie les valeurs manquantes et les valeurs invalides."""
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
        """Affiche le nombre moyen de vélos loués par heure."""

        # Calcule la moyenne des locations pour chaque heure.
        locations_par_heure = self._df.groupby("Hour")["Rented Bike Count"].mean()

        plt.plot(locations_par_heure.index, locations_par_heure.values)
        plt.xlabel("Heure de la journée")
        plt.ylabel("Nombre moyen de vélos loués")
        plt.title("Locations moyennes de vélos selon l'heure")
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.savefig("velo_loues_par_heure.png")
        plt.show()

    def __str__(self):
        return f"Dataset : {len(self._df)} lignes, {len(self._df.columns)} colonnes"

class Variable(ABC):
    # Classe parente des deux types de variables.
    def __init__(self, nom: str) -> None:
        self.nom = nom

    @abstractmethod
    def valider(self, texte: str):
        """Valide une valeur saisie par l'utilisateur."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.nom


class VariableNumerique(Variable):
    # Exemple : heure, humidité, température, vitesse du vent.
    def __init__(self, nom: str, minimum: float, maximum: float, entier: bool) -> None:
        super().__init__(nom)
        self.minimum = minimum
        self.maximum = maximum
        self.entier = entier

    def valider(self, texte: str):
        """Convertit une saisie en nombre et vérifie sa validité."""

        # Refuse un champ vide.
        if not texte.strip():
            raise DonneeInvalideError(f"{self.nom} : champ vide.")

        try:
            # Accepte aussi bien 2,5 que 2.5.
            valeur = float(texte.strip().replace(",", "."))
        except ValueError:
            raise DonneeInvalideError(f"{self.nom} : entre un nombre.")

        # Refuse NaN et les valeurs infinies.
        if not math.isfinite(valeur):
            raise DonneeInvalideError(f"{self.nom} : nombre invalide.")

        # Vérifie les limites autorisées.
        if valeur < self.minimum or valeur > self.maximum:
            raise DonneeInvalideError(
                f"{self.nom} : valeur entre {self.minimum:g} et {self.maximum:g}."
            )

        # L'heure doit par exemple être 17, et non 17.5.
        if self.entier:
            if not valeur.is_integer():
                raise DonneeInvalideError(f"{self.nom} : entier requis.")
            return int(valeur)

        return valeur


class VariableCategorielle(Variable):
    # Exemple : Summer, Winter, Holiday ou No Holiday.
    def __init__(self, nom: str, valeurs: list[str]) -> None:
        super().__init__(nom)
        self.valeurs = valeurs

    def valider(self, texte: str) -> str:
        """Accepte uniquement les catégories connues."""
        if texte not in self.valeurs:
            raise DonneeInvalideError(f"{self.nom} : choix invalide.")
        return texte

def enregistrer_metadonnees(df: pd.DataFrame, colonnes: list[str], r2: float) -> None:
    """Enregistre les variables et l'histogramme dans un fichier JSON."""
    variables = []

    # Décrit chacune des variables choisies pour entraîner le modèle.
    for nom in colonnes:
        serie = df[nom]

        if pd.api.types.is_numeric_dtype(serie):
            # Pour un nombre : minimum, maximum et type entier ou décimal.
            information = {
                "nom": nom,
                "type": "numerique",
                "minimum": float(serie.min()),
                "maximum": float(serie.max()),
                "entier": bool(pd.api.types.is_integer_dtype(serie))
            }
        else:
            # Pour une catégorie : liste des choix possibles.
            information = {
                "nom": nom,
                "type": "categorielle",
                "valeurs": sorted(serie.dropna().unique().tolist())
            }

        variables.append(information)

    # Compte le nombre d'heures dans chaque intervalle de locations.
    effectifs, bornes = np.histogram(df["Rented Bike Count"], bins=30)

    informations = {
        "variables": variables,
        "r2_test": float(r2),
        "histogramme": {
            "effectifs": effectifs.tolist(),
            "bornes": bornes.tolist()
        }
    }

    # Enregistre le JSON dans le dossier modeles du projet.
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dossier = os.path.join(racine, "modeles")
    os.makedirs(dossier, exist_ok=True)

    with open(os.path.join(dossier, "metadonnees.json"), "w", encoding="utf-8") as fichier:
        json.dump(informations, fichier, ensure_ascii=False, indent=2)

class Historique:
    def __init__(self, chemin: str) -> None:
        # Emplacement du fichier CSV qui conserve les prédictions.
        self._chemin = chemin

    def ajouter(self, valeurs: dict, prediction: int) -> None:
        """Ajoute une prédiction à la fin du fichier CSV."""
        dossier = os.path.dirname(self._chemin)
        if dossier:
            os.makedirs(dossier, exist_ok=True)

        # Les en-têtes sont ajoutés uniquement si le fichier est nouveau.
        nouveau_fichier = not os.path.isfile(self._chemin)

        # Prépare la date, les valeurs saisies et la prédiction.
        ligne = {"date_heure": datetime.now().isoformat(timespec="seconds")}
        ligne.update(valeurs)
        ligne["prediction"] = prediction

        # Le mode "a" ajoute une ligne sans effacer les anciennes.
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            writer = csv.DictWriter(fichier, fieldnames=ligne.keys())
            if nouveau_fichier:
                writer.writeheader()
            writer.writerow(ligne)

    def lire(self) -> list[dict]:
        """Retourne les anciennes prédictions sous forme de dictionnaires."""
        if not os.path.isfile(self._chemin):
            return []

        with open(self._chemin, "r", newline="", encoding="utf-8") as fichier:
            return list(csv.DictReader(fichier))