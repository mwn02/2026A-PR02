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
        """Garde le nom d'une variable et définit la méthode de validation que les deux types de variables devront utiliser avant une prédiction."""
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
        """Vérifie que le nombre entré par l'utilisateur respecte les limites de la variable avant de l'envoyer au modèle."""

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
    # Vérifie les choix catégoriels (ex. Summer, Winter, Holiday ou No Holiday).
    # Ces choix seront utilisés pour valider ce que l'utilisateur sélectionne dans l'interface.
    def __init__(self, nom: str, valeurs: list[str]) -> None:
        super().__init__(nom)
        self.valeurs = valeurs

    def valider(self, texte: str) -> str:
        """Accepte uniquement les catégories connues."""
        # texte : choix sélectionné par l'utilisateur (ex. "Summer" ou "No Holiday").
        # self.valeurs : liste des choix autorisés récupérés dans les métadonnées.
        if texte not in self.valeurs:
            raise DonneeInvalideError(f"{self.nom} : choix invalide.")
        return texte

def enregistrer_metadonnees(df: pd.DataFrame, colonnes: list[str], r2: float) -> None:
    """Enregistre les variables et l'histogramme dans un fichier JSON."""
    variables = []
    
    # Prépare les informations de chaque variable utilisée par le modèle.
    for nom in colonnes:
        serie = df[nom]
        # Si c'est une des deux variables catégorielles, on récupère ses choix
        # pour pouvoir les afficher plus tard dans l'interface.
        if nom == "Seasons" or nom == "Holiday":
            # Récupère les catégories sans les répéter.
            choix = []
            for valeur in serie:
                if valeur not in choix:
                    choix.append(valeur)

            choix.sort()

            information = {
                "nom": nom,
                "type": "categorielle",
                "valeurs": choix
            }

        else:
            # Toutes les autres variables sont numériques.
            # On conserve leurs limites pour vérifier les saisies dans l'interface.
            information = {
                "nom": nom,
                "type": "numerique",
                "minimum": float(min(serie)),
                "maximum": float(max(serie)),
                "entier": nom == "Hour" or nom == "Humidity(%)"
            }

        variables.append(information)


    
    # Répartit les nombres de vélos loués en 30 intervalles (ça aurait pu être un autre nombre).
    # effectifs : nombre d'heures observées dans chaque intervalle.
    # bornes : limites de chaque intervalle en nombre de vélos loués.
    effectifs_numpy, bornes_numpy = np.histogram(df["Rented Bike Count"], bins=30)

    # Convertit les tableaux NumPy en listes Python pour le fichier JSON.
    effectifs = effectifs_numpy.tolist()
    bornes = bornes_numpy.tolist()

    # Regroupe toutes les informations à enregistrer.
    informations = {
        "variables": variables,
        "r2_test": float(r2),
        "histogramme": {
            "effectifs": effectifs,
            "bornes": bornes
        }
    }

    
    # Récupère le chemin complet du fichier donnees.py.
    chemin_fichier = os.path.abspath(__file__)
    # Récupère le dossier qui contient donnees.py, donc src.
    dossier_src = os.path.dirname(chemin_fichier)
    # Remonte encore d'un dossier pour retrouver le dossier principal du projet.
    dossier_projet = os.path.dirname(dossier_src)
    # Ajoute "modeles" au chemin du projet pour choisir où enregistrer le JSON.
    dossier_modeles = os.path.join(dossier_projet, "modeles")
    # Crée le dossier modeles s'il n'existe pas déjà.
    os.makedirs(dossier_modeles, exist_ok=True)
    # Ajoute le nom du fichier JSON au chemin du dossier modeles.
    chemin_json = os.path.join(dossier_modeles, "metadonnees.json")
    with open(chemin_json, "w", encoding="utf-8") as fichier:
        json.dump(informations, fichier, ensure_ascii=False, indent=2)



class Historique:
    
    def __init__(self, chemin: str) -> None:
        # Garde le chemin du fichier CSV pour savoir où enregistrer les nouvelles prédictions et où retrouver les anciennes.
        self._chemin = chemin

    def ajouter(self, valeurs: dict, prediction: int) -> None:
        """Ajoute une prédiction à la fin du fichier CSV."""
        # Récupère le dossier dans lequel se trouve le fichier CSV.
        dossier = os.path.dirname(self._chemin)
        # Crée le dossier s'il n'existe pas déjà.
        if dossier:
            os.makedirs(dossier, exist_ok=True)

        # Les en-têtes sont ajoutés uniquement si le fichier est nouveau.
        nouveau_fichier = not os.path.isfile(self._chemin)

        # Récupère la date et l'heure actuelles pour identifier la prédiction.
        maintenant = datetime.now()
        # Convertit cette date en texte avec une précision à la seconde.
        date_heure = maintenant.isoformat(timespec="seconds")
        # Commence une nouvelle ligne avec la date et l'heure.
        ligne = {"date_heure": date_heure}

        # Ajoute les valeurs saisies par l'utilisateur (Hour, Seasons, etc.).
        ligne.update(valeurs)

        # Ajoute le nombre de vélos prédit dans la même ligne.
        ligne["prediction"] = prediction

        # Le mode "a" ajoute une ligne sans effacer les anciennes.
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            # Prépare l'écriture du dictionnaire sous forme de ligne CSV.
            writer = csv.DictWriter(fichier, fieldnames=ligne.keys())
            if nouveau_fichier:
                writer.writeheader()
            writer.writerow(ligne)

    
    def lire(self) -> list[dict]:
        """Récupère les prédictions déjà enregistrées dans le CSV
        pour pouvoir les afficher dans l'interface."""
        # Si le fichier CSV n'existe pas encore, il n'y a rien à afficher.
        if not os.path.isfile(self._chemin):
            return []
        # Ouvre le CSV pour récupérer toutes les prédictions enregistrées.
        with open(self._chemin, "r", newline="", encoding="utf-8") as fichier:
            # Lit les lignes sous forme de dictionnaires, puis les met dans une liste.
            return list(csv.DictReader(fichier))
