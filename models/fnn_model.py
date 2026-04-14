import torch
import torch.nn as nn

class OthelloFNN(nn.Module):
    """
    Réseau de Neurones Linéaire classique (FeedForward).
    Il prend les 64 cases du plateau comme un simple vecteur plat.
    """
    def __init__(self):
        super(OthelloFNN, self).__init__()
        
        # 64 entrées -> 128 neurones cachés -> 64 neurones -> 1 sortie (le score du plateau)
        self.reseau = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),           # Fonction d'activation pour casser la linéarité
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()            # Écrase la sortie entre -1 (défaite) et 1 (victoire)
        )

    def forward(self, x):
        # x est la grille. On s'assure qu'elle est aplatie (batch_size, 64)
        x = x.view(-1, 64) 
        return self.reseau(x)