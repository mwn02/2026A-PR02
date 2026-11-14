import pandas as pd
import pytest

from src.donnees import check_data_sanity

# Une fixture permet de définir les données une seule fois
# et de les réutiliser dans plusieurs tests.
@pytest.fixture
def donnees_valides():
    return pd.DataFrame({
        "Hour": [0, 23],
        "Humidity(%)": [0, 100],
        "Rented Bike Count": [0, 100],
        "Wind speed (m/s)": [0, 5],
        "Solar Radiation (MJ/m2)": [0, 2],
    })


def test_check_data_sanity_accepte_donnees_valides(donnees_valides):
    assert check_data_sanity(donnees_valides) is None

# permet l'exécution d'un test plusieurs fois avec des valeurs différentes
@pytest.mark.parametrize(
    "colonne, valeur, message",
    [
        ("Hour", float("nan"), "Valeurs manquantes dans le set de données."),
        ("Hour", -1, "Valeurs de l'heure hors de 0-23h."),
        ("Hour", 24, "Valeurs de l'heure hors de 0-23h."),
        ("Humidity(%)", -1, "Valeurs de l'humidité hors de 0-100%."),
        ("Humidity(%)", 101, "Valeurs de l'humidité hors de 0-100%."),
        ("Rented Bike Count", -1, "Valeurs du nombre de vélo loués sous 0."),
        ("Wind speed (m/s)", -1, "Valeurs de la vitesse du vent sous 0."),
        ("Solar Radiation (MJ/m2)", -1, "Valeurs des radiations solaires sous 0."),
    ],
)
def test_check_data_sanity_rejette_donnees_invalides(
    donnees_valides, colonne, valeur, message
):
    donnees_valides.loc[1, colonne] = valeur # modifie la valeur à la colonne donnée

    with pytest.raises(ValueError) as erreur:
        check_data_sanity(donnees_valides)

    assert str(erreur.value) == message # vérifie que le message d'erreur est celui attendu