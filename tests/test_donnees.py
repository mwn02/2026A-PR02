import pandas as pd
import pytest
from exceptions import DonneeInvalideError
from donnees import Dataset, VariableNumerique, VariableCategorielle, Historique

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
    assert Dataset.check_data_sanity(donnees_valides) is None

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
def test_check_data_sanity_rejette_donnees_invalides(donnees_valides, colonne, valeur, message):
    donnees_valides.loc[1, colonne] = valeur # modifie la valeur à la colonne donnée

    with pytest.raises(DonneeInvalideError) as erreur:
        Dataset.check_data_sanity(donnees_valides)

    assert str(erreur.value) == message # vérifie que le message d'erreur est celui attendu


def test_dataset_rejette_un_dossier(tmp_path):
    with pytest.raises(FileNotFoundError):
        Dataset(tmp_path)

def test_dataset_rejette_un_fichier_non_csv(tmp_path):
    fichier = tmp_path / "donnees.txt"
    fichier.touch() # crée le fichier temporaire vide

    with pytest.raises(ValueError, match="Le fichier de données doit être un fichier CSV."):
        Dataset(fichier)

def test_dataset_accepte_extension_csv_majuscule(tmp_path, donnees_valides):
    fichier = tmp_path / "donnees.CSV"
    donnees_valides.to_csv(fichier, index=False, encoding="cp1252")

    dataset = Dataset(fichier)
    pd.testing.assert_frame_equal(dataset.dataframe, donnees_valides)

def test_variable_numerique_accepte_heure_valide():
    # Crée une variable Hour qui accepte les entiers de 0 à 23.
    heure = VariableNumerique("Hour", 0, 23, True)

    # Vérifie que le texte "17" est converti en entier.
    assert heure.valider("17") == 17


@pytest.mark.parametrize(
    "saisie",
    ["-5", "24", "17.5", "abc", "", "nan"]
)
def test_variable_numerique_refuse_saisie_invalide(saisie):
    # Vérifie plusieurs erreurs : hors limites, décimale,
    # texte, champ vide et valeur non numérique valide.
    heure = VariableNumerique("Hour", 0, 23, True)

    # Chaque saisie invalide doit déclencher notre exception.
    with pytest.raises(DonneeInvalideError):
        heure.valider(saisie)


def test_variable_numerique_accepte_nombre_decimal():
    # Une variable décimale peut accepter une virgule.
    temperature = VariableNumerique(
        "Temperature(°C)", -20, 40, False
    )

    # pytest.approx compare les nombres décimaux.
    assert temperature.valider("20,5") == pytest.approx(20.5)


def test_variable_categorielle_accepte_saison_valide():
    # Définit les saisons disponibles.
    saison = VariableCategorielle(
        "Seasons",
        ["Winter", "Spring", "Summer", "Autumn"]
    )

    # Vérifie qu'une saison présente dans la liste est acceptée.
    assert saison.valider("Summer") == "Summer"


def test_variable_categorielle_refuse_saison_inconnue():
    saison = VariableCategorielle(
        "Seasons",
        ["Winter", "Summer"]
    )

    # Refuse une catégorie qui n'existe pas.
    with pytest.raises(DonneeInvalideError):
        saison.valider("Pluie")

def test_historique_enregistre_et_relit(tmp_path):
    # Crée un chemin temporaire.
    # On ne modifie pas le vrai historique de l'application.
    chemin = tmp_path / "historique" / "predictions.csv"

    historique = Historique(str(chemin))

    # Avant la première prédiction, l'historique est vide.
    assert historique.lire() == []

    # Enregistre deux prédictions fictives.
    historique.ajouter({"Hour": 17}, 1125)
    historique.ajouter({"Hour": 8}, 950)

    # Relit les prédictions depuis le fichier CSV.
    lignes = historique.lire()

    # Vérifie que les deux prédictions ont été conservées.
    assert len(lignes) == 2

    # Dans un CSV, les nombres sont relus comme du texte.
    assert lignes[0]["Hour"] == "17"
    assert lignes[0]["prediction"] == "1125"

    assert lignes[1]["Hour"] == "8"
    assert lignes[1]["prediction"] == "950"

    # Vérifie que chaque prédiction possède une date.
    assert "date_heure" in lignes[0]

