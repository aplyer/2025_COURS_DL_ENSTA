# TP 1 : Prise en main de l'ecosysteme de la Science des Donnees
#
# OBJECTIFS :
# 1. Se familiariser avec l'environnement de type notebook.
# 2. Revoir quelques concepts fondamentaux de Python.
# 3. Decouvrir NumPy pour le calcul numerique.
# 4. Decouvrir Matplotlib pour la visualisation de donnees.
# 5. Introduction a PyTorch et a la notion de calcul sur CPU vs. GPU.
#
# INSTRUCTIONS :
# Ce script est concu pour etre execute dans un environnement interactif comme
# Jupyter Notebook ou Google Colab. Chaque section peut etre copiee
# dans une cellule de code distincte. Les commentaires commencant par "##"
# sont des explications qui seraient idealement dans des cellules de texte (Markdown).

## --------------------------------------------------------------------------
## PARTIE 1 : RAPPELS DE PYTHON
## --------------------------------------------------------------------------
## Python est le langage de predilection en science des donnees. Sa syntaxe
## est simple et lisible. Un des aspects les plus importants est que les blocs
## de code (fonctions, boucles, etc.) sont definis par leur INDENTATION.

print("Bonjour le monde !")

# Definition d'une fonction
def carre(x):
    """Cette fonction retourne le carre de son argument."""
    return x ** 2

# Utilisation d'une boucle for pour iterer sur une liste
nombres = [1, 2, 3, 4, 5]
carres = []
for nombre in nombres:
    resultat = carre(nombre)
    carres.append(resultat)
    print(f"Le carre de {nombre} est {resultat}")

print("\nListe des carres :", carres)

# Python est aussi un langage oriente objet. On peut definir nos propres types de donnees.
class Point:
    """Une classe simple pour representer un point dans un plan 2D."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_origine(self):
        return (self.x**2 + self.y**2)**0.5

p = Point(3, 4)
print(f"\nLe point ({p.x}, {p.y}) est a une distance de {p.distance_origine()} de l'origine.")


## --------------------------------------------------------------------------
## PARTIE 2 : NUMPY - LE CALCUL SCIENTIFIQUE
## --------------------------------------------------------------------------
## NumPy est la bibliotheque fondamentale pour le calcul scientifique en Python.
## Elle fournit un objet puissant : le ndarray (tableau N-dimensionnel).
## Les operations sur les tableaux NumPy sont beaucoup plus rapides que sur
## les listes Python natives car elles sont implementees en C.

import numpy as np

# Creation d'un tableau NumPy a partir d'une liste Python
liste_py = [1, 2, 3, 4, 5]
array_np = np.array(liste_py)

print("\n--- NumPy ---")
print("Liste Python :", liste_py)
print("Tableau NumPy :", array_np)

# La grande difference : les operations vectorielles
liste_py_2 = liste_py + liste_py
array_np_2 = array_np + array_np

print("\nAddition d'une liste a elle-meme (concatenation) :", liste_py_2)
print("Addition d'un tableau a lui-meme (terme a terme) :", array_np_2)

# Creation de tableaux
zeros = np.zeros((2, 3)) # Un tableau 2x3 rempli de zeros
uns = np.ones((3, 2))   # Un tableau 3x2 rempli de uns
aleatoire = np.random.rand(2, 2) # Un tableau 2x2 avec des nombres aleatoires entre 0 et 1

print("\nTableau de zeros :\n", zeros)
print("Tableau de uns :\n", uns)
print("Tableau aleatoire :\n", aleatoire)

# Multiplication de matrices
mat_A = np.array([[1, 2], [3, 4]])
mat_B = np.array([[5, 6], [7, 8]])
produit_matriciel = mat_A @ mat_B # L'operateur @ est utilise pour le produit matriciel

print("\nProduit matriciel de A et B :\n", produit_matriciel)
print("Forme du tableau (shape) :", produit_matriciel.shape)


## --------------------------------------------------------------------------
## PARTIE 3 : MATPLOTLIB - LA VISUALISATION DE DONNEES
## --------------------------------------------------------------------------
## "Un bon croquis vaut mieux qu'un long discours". Matplotlib est la
## bibliotheque la plus utilisee pour creer des graphiques et des visualisations.

import matplotlib.pyplot as plt

# Creons quelques donnees avec NumPy
x = np.linspace(0, 2 * np.pi, 100) # 100 points entre 0 et 2*pi
y_sin = np.sin(x)
y_cos = np.cos(x)

# Creation du graphique
plt.figure(figsize=(10, 6)) # Definit la taille de la figure
plt.plot(x, y_sin, label='sin(x)', color='blue', linestyle='-')
plt.plot(x, y_cos, label='cos(x)', color='red', linestyle='--')

# Ajout de titres et legendes
plt.title("Fonctions Sinus et Cosinus")
plt.xlabel("Angle [rad]")
plt.ylabel("Valeur")
plt.grid(True)
plt.legend()

# Affichage du graphique
plt.show()


## --------------------------------------------------------------------------
## PARTIE 4 : PYTORCH - INTRODUCTION A L'APPRENTISSAGE PROFOND
## --------------------------------------------------------------------------
## PyTorch est l'un des frameworks d'apprentissage profond les plus populaires.
## Son objet de base est le Tenseur, qui est tres similaire au ndarray de NumPy,
## mais avec deux super-pouvoirs :
## 1. Il peut calculer automatiquement les gradients (pour la retropropagation).
## 2. Il peut effectuer des calculs sur des accelerateurs materiels (GPU, TPU).

import torch
import time

print("\n--- PyTorch ---")
print("Version de PyTorch :", torch.__version__)

# Creation d'un tenseur
tenseur = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("\nUn tenseur PyTorch :\n", tenseur)

# La question cruciale : ou sont mes donnees ? Sur quel "device" ?
print("Device du tenseur :", tenseur.device)

# Verifions si un GPU est disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nUn GPU est disponible ! Nous allons l'utiliser.")
else:
    device = torch.device("cpu")
    print("\nAucun GPU detecte. Utilisation du CPU.")

# Deplacons notre tenseur sur le device selectionne
tenseur_gpu = tenseur.to(device)
print("Device du nouveau tenseur :", tenseur_gpu.device)

# Tenter une operation entre un tenseur CPU et un tenseur GPU provoquera une erreur.
# C'est une erreur tres courante en pratique !
try:
    resultat_erreur = tenseur + tenseur_gpu
except RuntimeError as e:
    print("\nErreur attendue :", e)
    print("On ne peut pas operer sur des tenseurs qui ne sont pas sur le meme device.")

## --------------------------------------------------------------------------
## PARTIE 5 : EXERCICE DE SYNTHESE - LE GAIN DE PERFORMANCE DU GPU
## --------------------------------------------------------------------------
## Nous allons maintenant illustrer concretement l'interet du GPU pour les
## calculs massivement paralleles, comme la multiplication de grandes matrices.
## Nous allons chronometrer cette operation sur CPU puis sur GPU.

taille_matrice = 2000
iterations = 100

# --- Calcul sur CPU ---
print(f"\nDebut du calcul sur CPU (matrice {taille_matrice}x{taille_matrice}, {iterations} iterations)...")
mat_A_cpu = torch.randn(taille_matrice, taille_matrice, device='cpu')
mat_B_cpu = torch.randn(taille_matrice, taille_matrice, device='cpu')

start_time_cpu = time.time()
for _ in range(iterations):
    resultat_cpu = mat_A_cpu @ mat_B_cpu
end_time_cpu = time.time()

temps_cpu = end_time_cpu - start_time_cpu
print(f"Temps de calcul sur CPU : {temps_cpu:.4f} secondes.")

# --- Calcul sur GPU (si disponible) ---
if torch.cuda.is_available():
    print(f"\nDebut du calcul sur GPU (matrice {taille_matrice}x{taille_matrice}, {iterations} iterations)...")
    mat_A_gpu = mat_A_cpu.to(device)
    mat_B_gpu = mat_B_cpu.to(device)
    
    # Synchronisation pour une mesure de temps precise sur GPU
    # Les operations GPU sont asynchrones, il faut attendre qu'elles soient finies.
    torch.cuda.synchronize()
    
    start_time_gpu = time.time()
    for _ in range(iterations):
        resultat_gpu = mat_A_gpu @ mat_B_gpu
    torch.cuda.synchronize()
    end_time_gpu = time.time()

    temps_gpu = end_time_gpu - start_time_gpu
    print(f"Temps de calcul sur GPU : {temps_gpu:.4f} secondes.")
    
    print(f"\nLe calcul sur GPU etait environ {temps_cpu / temps_gpu:.2f} fois plus rapide !")

## --------------------------------------------------------------------------
## CONCLUSION DU TP
## --------------------------------------------------------------------------
## Dans ce TP, nous avons :
## - Revu les bases de Python.
## - Manipule des tableaux avec NumPy.
## - Cree un graphique simple avec Matplotlib.
## - Decouvert les tenseurs PyTorch et l'importance du "device" (CPU/GPU).
## - Demontre l'acceleration massive que peut apporter un GPU sur des taches
##   parallelisables.
##
## Vous etes maintenant prets a construire votre premier reseau de neurones !

