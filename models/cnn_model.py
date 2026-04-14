import torch
import torch.nn as nn

class OthelloCNN(nn.Module):
    """
    Réseau de Neurones Convolutif.
    Il regarde la grille comme une image 2D (8x8) pour comprendre la géométrie du jeu.
    """
    def __init__(self):
        super(OthelloCNN, self).__init__()
        
        # Les couches de convolution qui "scannent" le plateau
        self.convolutions = nn.Sequential(
            # Entrée : 1 canal (le plateau), Sortie : 32 filtres, Fenêtre de vue : 3x3
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # La tête de décision (on aplatit ce que les convolutions ont "vu" pour donner un score)
        self.decision = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Sortie entre -1 et 1
        )

    def forward(self, x):
        # x doit être au format "image" (batch_size, canaux, hauteur, largeur) -> (batch, 1, 8, 8)
        x = x.view(-1, 1, 8, 8)
        x = self.convolutions(x)
        
        # Aplatir pour passer dans la tête de décision
        x = x.view(x.size(0), -1) 
        return self.decision(x)