from donnees import load_data, box_plot_bike_count
from modele import preparer_donnees, separer_donnees, DummyRegressor

if __name__ == "__main__":
   df = load_data()
   box_plot_bike_count(df)
   # x récupère les informations utilisées pour prédire.
   # y récupère les réponses réelles : le nombre de vélos loués.
   x, y = preparer_donnees(df)
   # Récupère les données séparées : 80 % pour l'entraînement et 20 % pour le test.
   x_train, x_test, y_train, y_test = separer_donnees(x, y)
   # Passe les données d'entraînement à notre fonction et récupère le modèle entraîné dans reference.
   reference = DummyRegressor(x_train,y_train)
   