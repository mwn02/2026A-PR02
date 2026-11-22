import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modele import ApplicationVelos
from donnees import VariableCategorielle
from exceptions import DonneeInvalideError


# Associe les noms des colonnes CSV à des noms plus compréhensibles.
LIBELLES = {
    "Hour": "Heure",
    "Humidity(%)": "Humidité (%)",
    "Wind speed (m/s)": "Vent (m/s)",
    "Temperature(°C)": "Température (°C)",
    "Seasons": "Saison",
    "Holiday": "Jour férié ?",
    "Rainfall(mm)": "Précipitations (mm)"
}


def creer_interface():

    # Charge le modèle SGD entraîné et ses métadonnées JSON.
    # Aucun nouvel entraînement n'est nécessaire.
    application = ApplicationVelos()

    # Création de la fenêtre principale.
    fenetre = tk.Tk()
    fenetre.title("Prédiction de locations de vélos")
    fenetre.geometry("1000x600")
    fenetre.minsize(750, 450)

    # La colonne centrale occupe l'espace disponible.
    fenetre.columnconfigure(1, weight=1)
    fenetre.rowconfigure(0, weight=1)

    # ----- Formulaire de saisie -----

    formulaire = ttk.Frame(fenetre, padding=15)
    formulaire.grid(row=0, column=0, sticky="ns")

    # Conserve les champs pour récupérer leurs valeurs plus tard.
    champs = {}

    # Crée les sept champs à partir des métadonnées du modèle.
    for ligne, variable in enumerate(application.variables):

        nom = variable.nom

        # Affiche le nom de la variable en français.
        ttk.Label(
            formulaire,
            text=LIBELLES.get(nom, nom)
        ).grid(
            row=ligne * 2,
            column=0,
            sticky="w",
            pady=(0, 3)
        )

        if isinstance(variable, VariableCategorielle):

            # Liste déroulante pour les catégories.
            # Exemple : Summer, Winter, Holiday.
            champ = ttk.Combobox(
                formulaire,
                width=20,
                state="readonly",
                values=variable.valeurs
            )

        else:

            # Champ de texte pour les valeurs numériques.
            # Exemple : heure = 17, température = 20.
            champ = ttk.Entry(
                formulaire,
                width=20
            )

        # Place le champ sous son étiquette.
        champ.grid(
            row=ligne * 2 + 1,
            column=0,
            sticky="ew",
            pady=(0, 12)
        )

        # Conserve le champ pour récupérer sa valeur plus tard.
        champs[nom] = champ

    # ----- Colonne centrale -----

    centre = ttk.Frame(fenetre, padding=15)
    centre.grid(row=0, column=1, sticky="nsew")

    centre.columnconfigure(0, weight=1)
    centre.rowconfigure(1, weight=1)

    # Zone où apparaîtra le nombre de vélos prédit.
    prediction = ttk.LabelFrame(
        centre,
        text="Affichage de la prédiction",
        padding=15
    )

    prediction.grid(
        row=0,
        column=0,
        sticky="ew",
        pady=(0, 15)
    )

    
    # Crée un texte qui pourra changer après une prédiction. On le conserve dans une variable pour pouvoir le modifier.
    resultat = ttk.Label(
        prediction,
        text="La prédiction apparaîtra ici."
    )
    resultat.pack()


    # Zone où on ajoutera le graphique Matplotlib.
    graphique = ttk.LabelFrame(
        centre,
        text="Graphique",
        padding=15
    )

    graphique.grid(
        row=1,
        column=0,
        sticky="nsew"
    )

    
    # Récupère les données de l'histogramme enregistrées dans le JSON.
    histogramme = application.donnees_histogramme()

    bornes = histogramme["bornes"]
    effectifs = histogramme["effectifs"]

    # Calcule la largeur de chaque intervalle de locations.
    largeurs = []

    for i in range(len(effectifs)):
        largeur = bornes[i + 1] - bornes[i]
        largeurs.append(largeur)

    # Crée un graphique Matplotlib à l'intérieur de Tkinter.
    figure = Figure(figsize=(5, 4), dpi=95)
    axes = figure.add_subplot(111)

    # Les barres représentent les locations réellement observées.
    axes.bar(
        bornes[:-1],
        effectifs,
        width=largeurs,
        align="edge",
        color="steelblue",
        label="Locations observées"
    )

    # Prépare une ligne rouge pour afficher la prédiction.
    # Elle sera invisible jusqu'au clic sur le bouton "Prédire".
    ligne_prediction = axes.axvline(
        0,
        color="crimson",
        linewidth=2,
        label="Votre prédiction",
        visible=False
    )

    # Ajoute les titres et la légende du graphique.
    axes.set_title("Locations observées et prédiction")
    axes.set_xlabel("Nombre de vélos loués")
    axes.set_ylabel("Nombre d'heures observées")
    axes.legend()

    figure.tight_layout()

    # Insère le graphique Matplotlib dans la fenêtre Tkinter.
    canvas = FigureCanvasTkAgg(
        figure,
        master=graphique
    )

    canvas.get_tk_widget().pack(
        fill="both",
        expand=True
    )

    # Affiche le graphique.
    canvas.draw()


    # ----- Historique des prédictions -----

    historique = ttk.LabelFrame(
        fenetre,
        text="Historique",
        padding=10
    )

    historique.grid(
        row=0,
        column=2,
        sticky="ns",
        padx=(0, 15),
        pady=15
    )

    
    # Crée une liste qui affichera les anciennes prédictions.
    # On la conserve dans une variable pour pouvoir la modifier.
    liste_historique = tk.Listbox(
        historique,
        width=22
    )

    liste_historique.pack(
        fill="both",
        expand=True
    )

    # Recharge les prédictions enregistrées dans le fichier CSV.
    for ligne in application.lire_historique():

        liste_historique.insert(
            0,
            f"{ligne['date_heure'][5:16]} : "
            f"{ligne['prediction']} vélos"
        )

    # ----- Fonction exécutée lorsque l'utilisateur clique sur Prédire -----

    def au_clic_predire() -> None:
        """Récupère les saisies, prédit et actualise l'interface."""

        saisies = {}

        # Récupère les valeurs entrées dans les 7 champs.
        for nom, champ in champs.items():
            saisies[nom] = champ.get()

        try:
            # Envoie les valeurs au modèle SGD.
            # La méthode valide les saisies, fait la prédiction
            # et enregistre le résultat dans l'historique CSV.
            nombre = application.predire(saisies)

        except DonneeInvalideError as erreur:
            # Affiche un message si une valeur est invalide.
            messagebox.showerror("Saisie invalide", str(erreur))
            return

        except OSError as erreur:
            # Affiche un message si l'historique ne peut pas être enregistré.
            messagebox.showerror("Erreur d'enregistrement", str(erreur))
            return

        # Affiche le résultat prédit dans la fenêtre.
        resultat.config(
            text=f"Estimation : {nombre} vélos loués"
        )

        # Déplace la ligne rouge sur le nombre prédit.
        ligne_prediction.set_xdata([nombre, nombre])

        # Rend la ligne rouge visible.
        ligne_prediction.set_visible(True)

        # Ajuste le graphique si la valeur prédite dépasse les limites.
        axes.set_xlim(
            0,
            max(bornes[-1], nombre * 1.1, 1)
        )

        # Actualise le graphique.
        canvas.draw_idle()

        # Ajoute la nouvelle prédiction dans la liste historique.
        date = datetime.now().strftime("%m-%d %H:%M")

        liste_historique.insert(
            0,
            f"{date} : {nombre} vélos"
        )

    # Crée le bouton et le connecte à la fonction.
    bouton = ttk.Button(
        formulaire,
        text="Prédire",
        command=au_clic_predire
    )

    # Place le bouton sous les 7 champs.
    bouton.grid(
        row=len(application.variables) * 2,
        column=0,
        sticky="ew",
        pady=10
    )

    return fenetre


if __name__ == "__main__":
    creer_interface().mainloop()