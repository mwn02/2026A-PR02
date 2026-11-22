import os
import matplotlib.pyplot as plt
from donnees import Dataset, enregistrer_metadonnees
from modele import DummyModel, SGDModel
from sklearn.metrics import r2_score
import numpy as np


if __name__ == "__main__":
   path = os.path.join("donnees", "SeoulBikeData.csv")
   dataset = Dataset(path)
   dataset.plot_rented_bikes_per_hour()
   modele_dummy = DummyModel(dataset.dataframe)
   modele_sgd = SGDModel(dataset.dataframe)

   modele_dummy.entrainer()
   modele_sgd.entrainer()
   modele_dummy.save()
   modele_sgd.save()

   prediction_dummy = modele_dummy.predict(modele_dummy.x_test)
   prediction_sgd = modele_sgd.predict(modele_sgd.x_test)
   
   y_test = modele_sgd.y_test

   print("R² du modèle dummy :", round(r2_score(y_test, prediction_dummy), 4))
   print("R² du modèle SGD :", round(r2_score(y_test, prediction_sgd), 4))

   # Calcule la somme des erreurs au carré entre les valeurs
   # réellement observées et celles prédites par le modèle.
   erreurs_carre = np.sum((np.array(y_test) - np.array(prediction_sgd)) ** 2)
   # Calcule la somme des écarts au carré par rapport à la moyenne.
   # Cela sert de référence pour évaluer la qualité du modèle.
   ecarts_carre = np.sum((np.array(y_test) - np.mean(y_test)) ** 2)
   # Calcule le R² avec NumPy pour vérifier le résultat de scikit-learn.
   r2_numpy = 1 - erreurs_carre / ecarts_carre
   print("R² du modèle SGD (numpy) :", round(r2_numpy, 4))


   # Enregistre les informations dont l'interface a besoin :
   # les 7 variables, leurs limites, les catégories et le R².
   # Le JSON contient aussi les données de l'histogramme.
   enregistrer_metadonnees(dataset.dataframe, list(modele_sgd.x.columns), r2_score(y_test, prediction_sgd))

   # Compare les prédictions des deux modèles aux valeurs réelles du même jeu de test.
   plt.figure(figsize=(9, 6))
   plt.scatter(y_test, prediction_sgd, alpha=0.35, label="Modèle SGD")
   plt.scatter(y_test, prediction_dummy, alpha=0.35, label="Modèle dummy")
   minimum = min(prediction_sgd.min(), prediction_dummy.min())
   maximum = max(prediction_sgd.max(), prediction_dummy.max())
   plt.plot([minimum, maximum], [minimum, maximum], "k--", label="Prédiction parfaite") # k-- pour une ligne de traits noirs
   plt.xlabel("Nombre réel de vélos loués")
   plt.ylabel("Nombre prédit de vélos loués")
   plt.title("Prédictions du modèle SGD et du modèle dummy")
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig("predictions_modeles.png")
   plt.show()
