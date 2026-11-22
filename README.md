# Prédiction des locations de vélos à Séoul

Ce projet essaie de prédire combien de vélos seront loués pendant une heure à Séoul. Nous utilisons l'heure, la météo, la saison et les jours fériés pour entraîner un modèle de régression. Nous comparons ensuite ses prédictions à celles d'un modèle très simple qui prédit toujours la moyenne des locations.

## Données

Le fichier `donnees/SeoulBikeData.csv` vient du jeu de données [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), publié par Sathishkumar Veerappampalayam Easwaramoorthy sur l'UCI Machine Learning Repository (2020, [DOI : 10.24432/C5F62R](https://doi.org/10.24432/C5F62R)). Il contient 8 760 observations, soit une ligne par heure, et 14 colonnes. La valeur à prédire est `Rented Bike Count`.

Le fichier fait environ 0,6 Mo. Il est inclus dans le dépôt, car il est sous [licence CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) : sa redistribution est permise avec attribution. On peut aussi le télécharger depuis la page UCI ci-dessus et le placer dans `donnees/` sous le nom `SeoulBikeData.csv`.

## Installation et lancement

Il faut Python et un environnement graphique pour afficher les figures Matplotlib. Depuis la racine du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/main.py
```

Sur macOS ou Linux, la commande d'activation est `source .venv/bin/activate`. Le programme vérifie le CSV, entraîne les deux modèles, les enregistre dans `modeles/`, calcule leur R² et affiche deux graphiques. Les figures sont aussi enregistrées dans `velo_loues_par_heure.png` et `predictions_modeles.png`. Il enregistre ensuite dans `modeles/metadonnees.json` les informations nécessaires à l'interface et ouvre la fenêtre Tkinter.

On peut lancer l'interface seule avec `python src/interface.py` si `modeles/SGDModel.joblib` et `modeles/metadonnees.json` sont présents. Sinon, il faut d'abord exécuter `python src/main.py` pour les générer. L'interface valide les sept valeurs saisies, affiche une estimation et la situe sur l'histogramme des locations observées. Chaque prédiction est ajoutée à `historique/predictions.csv` et réaffichée au prochain lancement.

## Résultat du modèle

Les données sont séparées en 80 % pour l'entraînement et 20 % pour le test, avec `random_state=42`. Les deux modèles utilisent la même séparation. La métrique choisie est le R² : plus il est proche de 1, mieux les prédictions suivent les valeurs réelles; un score proche de 0 correspond au niveau du modèle qui prédit la moyenne.

| Modèle | R² sur le jeu de test |
| --- | ---: |
| Référence : `DummyRegressor`, qui prédit la moyenne | ≈ 0,000 |
| `SGDRegressor` | 0,473 |

Le modèle SGD fait donc mieux que cette référence, mais ses prédictions restent imparfaites. Ces scores ont été calculés avec le CSV du dépôt et le code actuel.

## Organisation du code

| Chemin | Rôle |
| --- | --- |
| `donnees/SeoulBikeData.csv` | Données utilisées par le projet |
| `src/donnees.py` | Chargement et vérification des données, graphique, validation des saisies, métadonnées et historique |
| `src/modele.py` | Préparation des variables, entraînement, prédiction, sauvegarde des modèles et décorateur `timer` |
| `src/main.py` | Analyse, comparaison des prédictions et lancement de l'interface |
| `src/interface.py` | Interface Tkinter de prédiction et consultation de l'historique |
| `tests/` | Tests des données, des validations, de l'historique et des modèles |
| `modeles/` | Modèles Joblib et métadonnées JSON utilisés par l'interface |
| `historique/predictions.csv` | Prédictions enregistrées par l'interface |

## Tests et documentation

Pour lancer les tests et mesurer la couverture :

```powershell
python -m coverage run -m pytest
python -m coverage report
```

Les 29 tests passent avec le code actuel. La commande `coverage report` affiche la couverture de l'exécution locale; elle ne mesure que les fichiers exécutés pendant les tests.

Les choix du projet sont expliqués dans [DECISIONS.md](DECISIONS.md) et l'utilisation de l'IA est consignée dans [JOURNAL_IA.md](JOURNAL_IA.md). Pour le fonctionnement des outils utilisés, voir aussi la documentation de [pandas](https://pandas.pydata.org/docs/), [scikit-learn](https://scikit-learn.org/stable/) et [pytest](https://docs.pytest.org/).

**Outil du chapitre 7 utilisé :** le décorateur `timer` défini dans `src/modele.py`. Il entoure les méthodes `entrainer()` des deux modèles, mesure leur durée avec `perf_counter` et affiche le temps écoulé en secondes.
