import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data() -> pd.DataFrame:
    """Vérifie que les données sont disponibles et valides. Les retournes dans une DataFrame."""
    df = load_raw_data()
    check_data_sanity(df)
    print("#### Dataset chargé. Aucune valeur aberrante. ####")
    return df

def load_raw_data() -> pd.DataFrame:
    """Vérifie que les données sont disponibles et les retourne en DataFrame"""
    path = os.path.join("donnees", "SeoulBikeData.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Le fichier de données est introuvable.")

    return pd.read_csv(path, encoding="cp1252")

def check_data_sanity(df: pd.DataFrame) -> None:
    """Vérifie qu'il n'y a aucune donnée manquante, ni de données ayant une valeur incorrect."""
    if (df.isnull().sum() > 0).any():
        raise ValueError("Valeurs manquantes dans le set de données.")
    if (df["Hour"] < 0).any() or (df["Hour"] > 23).any():
        raise ValueError("Valeurs de l'heure hors de 0-23h.")
    if (df["Humidity(%)"] < 0).any() or (df["Humidity(%)"] > 100).any():
        raise ValueError("Valeurs de l'humidité hors de 0-100%.")
    if (df["Rented Bike Count"] < 0).any():
        raise ValueError("Valeurs du nombre de vélo loués sous 0.")
    if (df["Wind speed (m/s)"] < 0).any():
        raise ValueError("Valeurs de la vitesse du vent sous 0.")
    if (df["Solar Radiation (MJ/m2)"] < 0).any():
        raise ValueError("Valeurs des radiations solaires sous 0.")

def rented_bikes_per_hour(df: pd.DataFrame) -> None:
    """Crée un graphique du nombre de vélos loués par heures."""
    # faire la moyenne des vélos loués par heure
    locations_par_heure = df.groupby("Hour")["Rented Bike Count"].mean()
    plt.plot(locations_par_heure.index, locations_par_heure.values)
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre moyen de vélos loués")
    plt.title("Locations moyennes de vélos selon l'heure")
    plt.xticks(range(0, 24)) # 24 heures
    plt.grid(True)
    plt.show()