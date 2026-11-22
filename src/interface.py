import tkinter as tk
from tkinter import ttk


def creer_interface():
    # création de la fenêtre
    fenetre = tk.Tk() # initialise l'application
    fenetre.title("Prédiction de locations de vélos")
    fenetre.geometry("1000x600")
    fenetre.minsize(750, 450) # taille minimum de la fenêtre

    fenetre.columnconfigure(1, weight=1) # colonne du milieu prend le plus d'espace
    fenetre.rowconfigure(0, weight=1)

    # entrées de données
    formulaire = ttk.Frame(fenetre, padding=15)
    formulaire.grid(row=0, column=0, sticky="ns") # colonne de gauche

    for ligne, nom in enumerate(("Heure", "Température (°C)", "Humidité (%)", "Saison", "Précipitations (mm)", "Vent (m/s)", "Jour de fête ?")):
        ttk.Label(formulaire, text=nom).grid(row=ligne * 2, column=0, sticky="w", pady=(0, 3))
        ttk.Entry(formulaire, width=20).grid(row=ligne * 2 + 1, column=0, sticky="ew", pady=(0, 12))

    ttk.Button(formulaire, text="Prédire").grid(row=16, column=0, sticky="ew", pady=10)

    # colonne centrale
    centre = ttk.Frame(fenetre, padding=15)
    centre.grid(row=0, column=1, sticky="nsew")
    centre.columnconfigure(0, weight=1)
    centre.rowconfigure(1, weight=1) # rangée du graphique

    # affichage de la prédiction
    prediction = ttk.LabelFrame(centre, text="Affichage de la prédiction", padding=15)
    prediction.grid(row=0, column=0, sticky="ew", pady=(0, 15))
    ttk.Label(prediction, text="La prédiction apparaîtra ici.").pack()

    # affichage du graphique
    graphique = ttk.LabelFrame(centre, text="Graphique", padding=15)
    graphique.grid(row=1, column=0, sticky="nsew")
    ttk.Label(graphique, text="Le graphique apparaîtra ici.").pack(expand=True)

    # tikitikitikiii tiki matakatikita
    historique = ttk.LabelFrame(fenetre, text="Historique", padding=10)
    historique.grid(row=0, column=2, sticky="ns", padx=(0, 15), pady=15)
    tk.Listbox(historique, width=22).pack(fill="both", expand=True)

    return fenetre


if __name__ == "__main__":
    creer_interface().mainloop()
