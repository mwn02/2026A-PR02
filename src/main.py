from donnees import load_data, box_plot_bike_count
from modele import preparer_donnees, separer_donnees, entrainer_dummy, entrainer_sgd

if __name__ == "__main__":
   # Charge les données avec donnees.py et les conserve dans df pour les réutiliser dans le reste du code
   df = load_data()
   # Affiche un box plot des nombres de vélos loués dans df pour repérer les valeurs inhabituelles.
   box_plot_bike_count(df)
   # Récupère les conditions/paramètres de df dans x et les nombres réels de vélos dans y.
   x, y = preparer_donnees(df)
   # Réserve 80 % des lignes et leurs réponses à l'entraînement, et 20 % à l'évaluation.
   x_train, x_test, y_train, y_test = separer_donnees(x, y)
   # Récupère le dummy qui prédit la moyenne de y_train pour comparer ensuite son erreur à celle de SGD.
   modele_dummy = entrainer_dummy(x_train, y_train)
   # Prédit cette moyenne pour chaque ligne réservée au test dans x_test.
   prediction_dummy = modele_dummy.predict(x_test)
   # Entraîne SGD sur les mêmes données et récupère le modèle avec sa standardisation, qui met les colonnes de x_train sur une même échelle.
   modele_sgd = entrainer_sgd(x_train, y_train)
   # Réutilise la standardisation apprise pour préparer x_test, puis prédit les vélos loués de chaque ligne.
   prediction_sgd = modele_sgd.predict(x_test)