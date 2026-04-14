# train.py
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim

# --- ASTUCE DE CHEMIN ---
# On ajoute le dossier parent au "chemin" de Python pour pouvoir importer 
# notre fichier othello_logic.py qui se trouve dans le dossier au-dessus.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import othello_logic as logic

from models.fnn_model import OthelloFNN
from models.cnn_model import OthelloCNN

def grille_vers_tenseur(Grille, joueur):
    """
    Traduit la grille (listes) en Tenseur PyTorch (matrice mathématique).
    Pour aider l'IA, on met le plateau de SON point de vue :
    1 = Mes pions, -1 = Pions adverses, 0 = Cases vides.
    """
    matrice = []
    joueur_adverse = (joueur + 1) % 2
    
    for x in range(8):
        ligne = []
        for y in range(8):
            if Grille[x][y] == joueur:
                ligne.append(1.0)
            elif Grille[x][y] == joueur_adverse:
                ligne.append(-1.0)
            else:
                ligne.append(0.0)
        matrice.append(ligne)
        
    return torch.tensor(matrice, dtype=torch.float32)

def jouer_partie_aleatoire():
    """
    Fait jouer deux bots aléatoires et sauvegarde l'historique des plateaux.
    """
    Grille = logic.initGrille()
    auTour = 0
    historique_etats = [] # Sauvegardera les grilles
    historique_joueurs = [] # Sauvegardera à qui c'était le tour
    
    Cases_jouables = logic.get_cases_jouables(Grille, auTour)
    
    while True:
        if len(Cases_jouables) == 0:
            auTour += 1
            Cases_jouables = logic.get_cases_jouables(Grille, auTour)
            if len(Cases_jouables) == 0:
                break # Fin de la partie
            continue # Passe le tour
            
        # Choix aléatoire (Pour l'instant, l'IA explore)
        x, y = random.choice(list(Cases_jouables.keys()))
        
        # On sauvegarde le plateau AVANT de jouer, de la perspective du joueur
        etat_tenseur = grille_vers_tenseur(Grille, auTour % 2)
        historique_etats.append(etat_tenseur)
        historique_joueurs.append(auTour % 2)
        
        # On joue le coup
        logic.renversement(Grille, Cases_jouables, x, y, auTour)
        logic.coup(Grille, x - 1, y - 1, auTour % 2)
        
        auTour += 1
        Cases_jouables = logic.get_cases_jouables(Grille, auTour)
        
    # Calcul des récompenses finales (1 pour victoire, -1 pour défaite)
    blanc, noir = logic.score(Grille)
    recompenses = []
    
    for joueur in historique_joueurs:
        if joueur == 0: # Noir
            if noir > blanc: recompenses.append(1.0)
            elif blanc > noir: recompenses.append(-1.0)
            else: recompenses.append(0.0)
        else: # Blanc
            if blanc > noir: recompenses.append(1.0)
            elif noir > blanc: recompenses.append(-1.0)
            else: recompenses.append(0.0)
            
    return historique_etats, recompenses

def entrainer_modele(modele, nom_fichier_sauvegarde, nb_parties=1000):
    """
    La boucle d'entraînement principale.
    """
    print(f"--- Début de l'entraînement de {nom_fichier_sauvegarde} ---")
    
    # L'optimiseur "Adam" est l'algorithme qui va modifier les poids du réseau
    optimiseur = optim.Adam(modele.parameters(), lr=0.001)
    # La fonction de perte (Mean Squared Error) pour calculer l'erreur
    critere = nn.MSELoss()
    
    modele.train() # Met le modèle en mode apprentissage
    
    for i in range(nb_parties):
        # 1. On génère une partie
        etats, recompenses = jouer_partie_aleatoire()
        
        if len(etats) == 0: continue
            
        # 2. On convertit nos listes en gros paquets (batch) pour PyTorch
        batch_etats = torch.stack(etats)
        batch_recompenses = torch.tensor(recompenses, dtype=torch.float32).view(-1, 1)
        
        # 3. Le modèle essaie de deviner le score final pour chaque plateau
        predictions = modele(batch_etats)
        
        # 4. On calcule l'erreur entre sa prédiction et la vraie fin de partie
        perte = critere(predictions, batch_recompenses)
        
        # 5. La magie des Maths (Rétropropagation du gradient)
        optimiseur.zero_grad() # On remet à zéro
        perte.backward()       # On calcule dans quel sens modifier les neurones
        optimiseur.step()      # On modifie les neurones !
        
        # Affichage de la progression
        if (i + 1) % 100 == 0:
            print(f"Partie {i + 1}/{nb_parties} | Erreur moyenne (Loss) : {perte.item():.4f}")
            
    # Sauvegarde du cerveau dans un fichier !
    torch.save(modele.state_dict(), nom_fichier_sauvegarde)
    print(f"Entraînement terminé. Cerveau sauvegardé sous '{nom_fichier_sauvegarde}'")

if __name__ == "__main__":
    # On crée nos deux modèles "vides"
    reseau_dense = OthelloFNN()
    reseau_convolutif = OthelloCNN()
    
    # On les entraîne sur 1000 parties chacun (tu pourras augmenter ce chiffre !)
    print("Entraînement du modèle Basique (FNN)...")
    entrainer_modele(reseau_dense, "fnn_cerveau.pth", nb_parties=1000)
    
    print("\nEntraînement du modèle Avancé (CNN)...")
    entrainer_modele(reseau_convolutif, "cnn_cerveau.pth", nb_parties=1000)