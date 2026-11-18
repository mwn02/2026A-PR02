from sklearn.model_selection import train_test_split
def preparer_donnees(df):
    # prend du dataframe les variables suivantes
    x = df[["Hour", "Humidity(%)", "Humidity(%)", "Wind speed (m/s)", "Solar Radiation (MJ/m2)", ]]
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
    return x_train,x_test,y_train,y_train