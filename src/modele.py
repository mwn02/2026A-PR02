
def preparer_donnees(df):
    # prend du dataframe les variables suivantes
    x = df[["Hour", "Humidity(%)", "Humidity(%)", "Wind speed (m/s)", "Solar Radiation (MJ/m2)", ]]
    # prend du dataframe les réponses à prédire
    y = df["Rented Bike Count"]
    # Renvoie deux éléments dans cet ordre : 1. le tableau des informations. 2. la colonne des réponses.
    return x,y