import os
import matplotlib.pyplot as plt
from donnees import Dataset
from modele import DummyModel, SGDModel

if __name__ == "__main__":
   path = os.path.join("donnees", "SeoulBikeData.csv")
   dataset = Dataset(path)
   dataset.plot_rented_bikes_per_hour()
   modele_dummy = DummyModel(dataset.dataframe)
   modele_sgd = SGDModel(dataset.dataframe)

   modele_dummy.entrainer()
   modele_sgd.entrainer()

   prediction_dummy = modele_dummy.predict(modele_dummy.x_test)
   prediction_sgd = modele_sgd.predict(modele_sgd.x_test)
   
   y_test = modele_sgd.y_test

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
