# -*- coding: utf-8 -*-
"""
TP3 : Optimiseurs et Retropropagation Manuelle

Objectifs :
1.  Partie 1 : Comparer empiriquement les performances des optimiseurs
    SGD (avec momentum), RMSProp et Adam sur un probleme de classification non-lineaire.
2.  Partie 2 : Comprendre la mecanique de la retropropagation en calculant
    "a la main" les gradients d'un petit MLP et en les verifiant avec autograd.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration Generale ---
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# --- PARTIE 1 : COMPARAISON DES OPTIMISEURS ---
# ==============================================================================

print("--- Debut de la Partie 1 : Comparaison des Optimiseurs ---")

# --- 1.1 Preparation des donnees ---
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# --- 1.2 Modele et fonction d'entrainement ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_with_optimizer(optimizer_class, **kwargs):
    """Entraine un modele avec un optimiseur donne et retourne l'historique des pertes."""
    model = SimpleMLP()
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            history.append(loss.item())
    return history

# --- 1.3 Lancement des entrainements ---
lr = 0.01
print(f"Lancement des entrainements avec un learning rate de {lr}...")
history_sgd = train_with_optimizer(optim.SGD, lr=lr, momentum=0.9)
history_rmsprop = train_with_optimizer(optim.RMSprop, lr=lr)
history_adam = train_with_optimizer(optim.Adam, lr=lr)
print("Entrainements termines.")

# --- 1.4 Visualisation ---
plt.figure(figsize=(10, 6))
plt.plot(history_sgd, label='SGD with Momentum', color='blue', linestyle='--')
plt.plot(history_rmsprop, label='RMSProp', color='green', linestyle='-.')
plt.plot(history_adam, label='Adam', color='red', linestyle='-')
plt.title('Comparaison des Courbes de Perte des Optimiseurs')
plt.xlabel('Iterations (x5 epoques)')
plt.ylabel('Perte d\'entrainement (Entropie Croisee)')
plt.legend()
plt.grid(True)
plt.ylim(0, 0.8)
plt.show()

# ==============================================================================
# --- PARTIE 2 : RETROPROPAGATION MANUELLE ---
# ==============================================================================

print("\n--- Debut de la Partie 2 : Retropropagation Manuelle ---")

# --- 2.1 Definition d'un MLP simple et de donnees jouets ---
# Un seul point de donnee pour simplifier le calcul
x_sample = torch.tensor([[2.0, 3.0]])
y_sample = torch.tensor([[1.0]])

# Modele : 2 entrees -> 3 neurones caches -> 1 sortie
# On definit les poids et biais explicitement pour y acceder
W1 = torch.randn(2, 3, requires_grad=True)
b1 = torch.randn(3, requires_grad=True)
W2 = torch.randn(3, 1, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)

# Activation et perte
relu = nn.ReLU()
mse_loss = nn.MSELoss()

# --- 2.2 Forward Pass et calcul de la perte ---
# Couche 1
z1 = x_sample @ W1 + b1
a1 = relu(z1)
# Couche 2 (sortie)
z2 = a1 @ W2 + b2
# Pas d'activation sur la sortie pour une regression simple
y_pred = z2
# Perte
loss = mse_loss(y_pred, y_sample)
print(f"Perte initiale: {loss.item()}")

# --- 2.3 Calcul des gradients avec Autograd (notre reference) ---
loss.backward()
print("\n--- Gradients calcules par Autograd (reference) ---")
print("Gradient de W2:\n", W2.grad)
print("Gradient de b2:\n", b2.grad)
print("Gradient de W1:\n", W1.grad)
print("Gradient de b1:\n", b1.grad)

# --- 2.4 Calcul des gradients manuellement ---
# On remet les gradients a zero avant de recalculer
W1.grad.zero_()
b1.grad.zero_()
W2.grad.zero_()
b2.grad.zero_()

print("\n--- Calcul manuel des gradients (Backward Pass) ---")
# 1. Derivee de la perte par rapport a la prediction
# Perte L = 1/2 * (y_pred - y)^2  => dL/dy_pred = y_pred - y
grad_y_pred = y_pred - y_sample

# 2. Gradients de la couche de sortie (W2, b2)
# y_pred = z2, donc dL/dz2 = dL/dy_pred * dy_pred/dz2 = grad_y_pred * 1
grad_z2 = grad_y_pred

# dL/dW2 = dL/dz2 * dz2/dW2 = grad_z2 * a1
# Les dimensions doivent correspondre : (3,1) = (1,3).T @ (1,1)
grad_W2 = a1.T @ grad_z2
# dL/db2 = dL/dz2 * dz2/db2 = grad_z2 * 1
grad_b2 = grad_z2

# 3. Propagation vers la couche cachee
# dL/da1 = dL/dz2 * dz2/da1 = grad_z2 * W2
# Les dimensions doivent correspondre : (1,3) = (1,1) @ (1,3)
grad_a1 = grad_z2 @ W2.T

# 4. Gradients de la couche cachee (W1, b1)
# dL/dz1 = dL/da1 * da1/dz1 = grad_a1 * relu'(z1)
# La derivee de ReLU(z) est 1 si z > 0, 0 sinon.
grad_relu_z1 = (z1 > 0).float()
grad_z1 = grad_a1 * grad_relu_z1

# dL/dW1 = dL/dz1 * dz1/dW1 = grad_z1 * x_sample
# Les dimensions doivent correspondre : (2,3) = (1,2).T @ (1,3)
grad_W1 = x_sample.T @ grad_z1
# dL/db1 = dL/dz1 * dz1/db1 = grad_z1 * 1
grad_b1 = grad_z1

print("Gradient manuel de W2:\n", grad_W2)
print("Gradient manuel de b2:\n", grad_b2)
print("Gradient manuel de W1:\n", grad_W1)
print("Gradient manuel de b1:\n", grad_b1.squeeze()) # Squeeze pour enlever la dim superflue

# --- 2.5 Verification ---
print("\n--- Verification des differences ---")
print(f"Difference max sur W2: {torch.max(torch.abs(W2.grad - grad_W2))}")
print(f"Difference max sur b2: {torch.max(torch.abs(b2.grad - grad_b2))}")
print(f"Difference max sur W1: {torch.max(torch.abs(W1.grad - grad_W1))}")
print(f"Difference max sur b1: {torch.max(torch.abs(b1.grad - grad_b1.squeeze()))}")
print("\nSi les differences sont tres faibles (proches de 0), nos calculs sont corrects !")

