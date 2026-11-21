# Décisions

| Choix | Options envisagées | Choix retenu et justification | Validé par l'auxiliaire (date) |
| --- | --- | --- | --- |
| Domaine | Sport, Transport | Transport<br>Justification : Prédire la demande de vélos selon la météo et l'heure | 8 novembre 2026 |
| Jeu de données | Gym Members, Seoul Bike Sharing Demand | Seoul Bike Sharing Demand<br>Source : UCI Machine Learning Repository<br>Taille : 8760 observations<br>Licence : CC By 4.0<br>Justification : Données suffisantes et variables adaptées à notre projet | 8 novembre 2026 |
| Variable cible et métrique | (options pour la métrique)<br>- Erreur absolue moyenne<br>- R^2| Cible : Nombre de vélos loués par heure<br>Type : Régression (on prédit une quantité)<br>Métrique : R^2, puisque nous prédisons une quantité. Cette métrique nous permet donc de voir à quel point les prédictions sont proches des vrais valeurs et à quel point le modèle est meilleur qu'un modèle trivial prédisant une simple moyenne.| 20 novemebre 2026|
| Modèle | - Régression par descente de gradient stochastique ( SGD Regressor)<br>- K plus proches voisins (KNN) | SGD Regressor<br>Justification : C'est un modèle fait pour prédire une quantité, dont le nombre de données du dataset est inférieur à 100 000 (ce qui est notre cas). Nous nous sommes basés sur https://scikit-learn.org/stable/machine_learning_map.html.| 20 novembre 2026|
| Interface graphique | - Tkinter<br>- PyQt  | Tkinter: c'est simple et c'est déjà un module de python.    | 20 novembre 2026|
