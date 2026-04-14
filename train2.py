# train.py
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import othello_logic as logic

from models.fnn_model import OthelloFNN
from models.cnn_model import OthelloCNN

def grille_vers_tenseur(Grille, joueur):
    """Traduit la grille pour le réseau : 1 = Moi, -1 = Adversaire, 0 = Vide."""
    matrice = []
    joueur_adverse = (joueur + 1) % 2
    for x in range(8):
        ligne = []
        for y in range(8):
            if Grille[x][y] == joueur: ligne.append(1.0)
            elif Grille[x][y] == joueur_adverse: ligne.append(-1.0)
            else: ligne.append(0.0)
        matrice.append(ligne)
    return torch.tensor(matrice, dtype=torch.float32)

def choisir_coup_epsilon_greedy(modele, Grille, Cases_jouables, joueur, epsilon):
    """
    Le cœur de l'apprentissage ! Choisit entre l'Exploration (hasard) 
    et l'Exploitation (utilisation du cerveau).
    """
    # EXPLORATION : On tire un nombre au hasard. S'il est sous epsilon, on joue au pif.
    if random.random() < epsilon:
        return random.choice(list(Cases_jouables.keys()))
        
    # EXPLOITATION : On utilise le réseau de neurones pour évaluer tous les futurs possibles.
    meilleur_score = float('-inf')
    meilleurs_coups = []
    
    # On met le modèle en mode "évaluation" (désactive temporairement l'apprentissage)
    modele.eval()
    
    # On n'a pas besoin de calculer les gradients (gain de mémoire et de vitesse)
    with torch.no_grad():
        for (x, y) in Cases_jouables:
            # 1. On simule le coup
            grille_simulee = logic.initGrille()
            logic.copy_grille(Grille, grille_simulee)
            logic.renversement(grille_simulee, Cases_jouables, x, y, joueur)
            logic.coup(grille_simulee, x - 1, y - 1, joueur)
            
            # 2. On traduit la grille pour le modèle (toujours du point de vue du joueur)
            etat_tenseur = grille_vers_tenseur(grille_simulee, joueur)
            
            # 3. Le modèle donne une note à ce plateau
            score = modele(etat_tenseur).item()
            
            if score > meilleur_score:
                meilleur_score = score
                meilleurs_coups = [(x, y)]
            elif score == meilleur_score:
                meilleurs_coups.append((x, y))
                
    modele.train() # On remet le modèle en mode "apprentissage"
    
    # S'il y a plusieurs coups à égalité, on en prend un au hasard parmi les meilleurs
    return random.choice(meilleurs_coups)

def jouer_partie(modele, epsilon):
    """Joue une partie en utilisant la stratégie epsilon-greedy."""
    Grille = logic.initGrille()
    auTour = 0
    historique_etats = []
    historique_joueurs = []
    
    Cases_jouables = logic.get_cases_jouables(Grille, auTour)
    
    while True:
        if len(Cases_jouables) == 0:
            auTour += 1
            Cases_jouables = logic.get_cases_jouables(Grille, auTour)
            if len(Cases_jouables) == 0:
                break
            continue
            
        # L'IA choisit son coup avec Epsilon-Greedy !
        x, y = choisir_coup_epsilon_greedy(modele, Grille, Cases_jouables, auTour % 2, epsilon)
        
        etat_tenseur = grille_vers_tenseur(Grille, auTour % 2)
        historique_etats.append(etat_tenseur)
        historique_joueurs.append(auTour % 2)
        
        logic.renversement(Grille, Cases_jouables, x, y, auTour)
        logic.coup(Grille, x - 1, y - 1, auTour % 2)
        
        auTour += 1
        Cases_jouables = logic.get_cases_jouables(Grille, auTour)
        
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

def entrainer_modele(modele, nom_fichier_sauvegarde, nb_parties=2000):
    print(f"\n--- Début de l'entraînement de {nom_fichier_sauvegarde} ---")
    
    optimiseur = optim.Adam(modele.parameters(), lr=0.001)
    critere = nn.MSELoss()
    
    # Paramètres de l'Epsilon-Greedy
    epsilon = 1.0          # Commence à 100% de hasard
    epsilon_min = 0.05     # Ne descend jamais sous 5% de hasard
    epsilon_decay = 0.998  # Multiplicateur à chaque partie
    
    modele.train()
    
    for i in range(nb_parties):
        # On joue la partie avec le taux de hasard actuel
        etats, recompenses = jouer_partie(modele, epsilon)
        
        if len(etats) == 0: continue
            
        batch_etats = torch.stack(etats)
        batch_recompenses = torch.tensor(recompenses, dtype=torch.float32).view(-1, 1)
        
        predictions = modele(batch_etats)
        perte = critere(predictions, batch_recompenses)
        
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        
        # On réduit le hasard pour la prochaine partie !
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        if (i + 1) % 100 == 0:
            print(f"Partie {i + 1:04d}/{nb_parties} | Hasard: {epsilon*100:.1f}% | Erreur: {perte.item():.4f}")
            
    torch.save(modele.state_dict(), nom_fichier_sauvegarde)
    print(f"Cerveau sauvegardé sous '{nom_fichier_sauvegarde}'")

if __name__ == "__main__":
    reseau_dense = OthelloFNN()
    reseau_convolutif = OthelloCNN()
    
    # Pour un bon apprentissage avec Epsilon, il faut plus de parties (ex: 2000)
    entrainer_modele(reseau_dense, "fnn_cerveau.pth", nb_parties=2000)
    entrainer_modele(reseau_convolutif, "cnn_cerveau.pth", nb_parties=2000)