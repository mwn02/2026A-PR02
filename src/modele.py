from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def preparer_donnees(df):
    # Sélectionne dans df, reçu du main, les conditions utilisées pour prédire les locations.
    x = df[["Hour", "Humidity(%)", "Wind speed (m/s)", "Solar Radiation (MJ/m2)"]]
    # Récupère les nombres réels de vélos pour apprendre, puis vérifier les prédictions.
    y = df["Rented Bike Count"]
    # Renvoie les conditions et leurs réponses dans les variables x et y du main.
    return x, y

def separer_donnees(x, y):
    # Répartit les lignes en 80 % d'entraînement et 20 % de test.
    # random_state=42 refait le même mélange à chaque lancement, pour comparer les modèles avec les mêmes lignes d'entraînement et de test.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    # Renvoie au main les conditions d'entraînement et de test, puis leurs réponses.
    return x_train, x_test, y_train, y_test


def entrainer_dummy(x_train, y_train):
    # Crée le dummy qui prédira toujours la moyenne : son erreur servira de repère, et on attend de SGD une erreur plus petite.
    reference = DummyRegressor(strategy="mean")
    # fit mémorise la moyenne de y_train (sera utile pour predit() dans le main ; x_train est fourni, mais ses valeurs sont ignorées.
    reference.fit(x_train, y_train)
    # Renvoie le modèle entraîné au main.
    return reference


def entrainer_sgd(x_train, y_train):
    # Crée l'outil qui mettra les colonnes de x_train à des échelles comparables pour aider SGD à apprendre(ou sinon peut pas, ex: vent => [0-200] et humidite => [0,100]).
    echelle = StandardScaler()
    # Réunit cet outil et SGD pour préparer les données avant l'entraînement et les prédictions.
    # random_state=42 permet de refaire le même entraînement avec les mêmes données.
    modele_sgd = make_pipeline(echelle, SGDRegressor(random_state=42))
    # fit() mémorise la standardisation sur x_train, puis entraîne SGD avec les conditions préparées et y_train.
    modele_sgd.fit(x_train, y_train)
    # Renvoie au main la standardisation et SGD entraînés, réunis pour pouvoir prédire sur x_test.
    return modele_sgd