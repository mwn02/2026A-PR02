from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
def preparer_donnees(df):
    # prend du dataframe les variables suivantes
    x = df[["Hour", "Humidity(%)", "Wind speed (m/s)", "Solar Radiation (MJ/m2)", ]]
    # prend du dataframe les réponses à prédire
    y = df["Rented Bike Count"]
    # Renvoie deux éléments dans cet ordre : 1. le tableau des informations. 2. la colonne des réponses.
    return x,y

def separer_donnees(x,y):
    #x : infromation pour predire
    #y : resultat attendu
    # Répartit les observations : 80 % pour apprendre, 20 % pour évaluer.
    # Les informations et leurs réponses restent associées.
    # random_state : Garde les mêmes exemples d'entraînement et de test à chaque exécution
    # pour comparer les modèles sans que les résultats changent à cause d'une nouvelle répartition.
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)
    return x_train,x_test,y_train,y_test

def entrainer_dummy(x_train, y_train):
    # Crée un modèle configuré pour prédire toujours la moyenne, sans encore la calculer.
    reference = DummyRegressor(strategy="mean")
    # fit calcule et mémorise la moyenne de y_train ; x_train important, mais il ne s'en sert pas.
    reference.fit(x_train, y_train)
    # Renvoie le modèle entraîné, qui pourra utiliser cette moyenne pour faire ses prédictions.
    return reference
