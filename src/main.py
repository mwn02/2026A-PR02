import os
from donnees import Dataset
from modele import preparer_donnees, separer_donnees, entrainer_dummy

if __name__ == "__main__":
   path = os.path.join("donnees", "SeoulBikeData.csv")
   dataset = Dataset(path)
   dataset.plot_rented_bikes_per_hour()
   # x récupère les informations utilisées pour prédire.
   # y récupère les réponses réelles : le nombre de vélos loués.
   x, y = preparer_donnees(dataset.dataframe)
   # Récupère les données séparées : 80 % pour l'entraînement et 20 % pour le test.
   x_train, x_test, y_train, y_test = separer_donnees(x, y)
   # Passe les données d'entraînement à notre fonction et récupère le modèle entraîné dans reference.
   reference = entrainer_dummy(x_train,y_train)
   prediction = reference.predict(x_test)