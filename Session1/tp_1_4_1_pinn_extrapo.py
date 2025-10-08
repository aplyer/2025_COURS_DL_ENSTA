# -*- coding: utf-8 -*-
"""
TP Cas Intermediaire : Regression, Extrapolation et PINNs

Objectifs :
1.  Observer l'echec d'un MLP en extrapolation lorsqu'il est entraine
    uniquement sur des donnees (regression pure).
2.  Visualiser l'impact du bruit sur l'overfitting et l'extrapolation.
3.  Demontrer comment l'ajout d'une perte physique (PINN) contraint
    la solution et permet une extrapolation correcte, meme avec des
    donnees bruitees.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Configuration Generale et Outils ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {DEVICE}")

# Parametres physiques de l'oscillateur
m, mu, k = 1.0, 2.0, 100.0
d, w0 = mu / (2 * m), np.sqrt(k / m)

# Solution analytique pour reference
def exact_solution(d_val, w0_val, t):
    """Calcule la solution analytique de l'oscillateur harmonique amorti."""
    w = np.sqrt(w0_val**2 - d_val**2)
    phi = np.arctan(-d_val / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d_val * t)
    return exp * 2 * A * cos

# Architecture du reseau de neurones (MLP)
class FCN(nn.Module):
    """Reseau de neurones a connexion totale (MLP)."""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN), activation(),
            *[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS-1)],
            nn.Linear(N_HIDDEN, N_OUTPUT)
        )
    def forward(self, x):
        return self.net(x)

# --- 2. Preparation des Domaines et Donnees ---

# Domaine temporel complet pour le test/visualisation
t_full_domain = torch.linspace(0, 1, 300).view(-1, 1).to(DEVICE)
u_exact_full = exact_solution(d, w0, t_full_domain.cpu()).to(DEVICE)

# Domaine d'entrainement restreint (interpolation)
t_train_limit = 0.5
N_train_data = 60
t_data_train = torch.linspace(0, t_train_limit, N_train_data).view(-1, 1).to(DEVICE)

# Donnees d'entrainement sans bruit
u_data_train_noiseless = exact_solution(d, w0, t_data_train.cpu()).to(DEVICE)

# Donnees d'entrainement avec bruit
noise_level = 0.05
u_data_train_noisy = u_data_train_noiseless + noise_level * torch.randn_like(u_data_train_noiseless)

# Points de collocation pour la perte physique (sur TOUT le domaine)
N_physics = 120
t_physics = torch.linspace(0, 1, N_physics, requires_grad=True).view(-1, 1).to(DEVICE)


# --- 3. Definition des Boucles d'Entrainement ---

def train_regression_pure(data_train, n_iter=15001):
    """Entraine un MLP en regression pure sur les donnees fournies."""
    model = FCN(1, 1, 32, 3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    for i in range(n_iter):
        optimizer.zero_grad()
        u_pred = model(data_train[0]) # t_data_train
        loss = criterion(u_pred, data_train[1]) # u_data_train
        loss.backward()
        optimizer.step()
        if i % 3000 == 0:
            print(f"Iteration {i}, Perte (MSE): {loss.item():.4e}")
    print(f"Entrainement termine en {time.time() - start_time:.2f}s")
    return model

def train_pinn(data_train, n_iter=15001):
    """Entraine un PINN avec perte de donnees et perte physique."""
    model = FCN(1, 1, 32, 3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_data = nn.MSELoss()
    lambda_phys = 1e-4

    start_time = time.time()
    for i in range(n_iter):
        optimizer.zero_grad()

        # Perte sur les donnees (domaine restreint)
        u_pred_data = model(data_train[0])
        loss_data = criterion_data(u_pred_data, data_train[1])

        # Perte physique (domaine complet)
        u_phys = model(t_physics)
        dudt = torch.autograd.grad(u_phys.sum(), t_physics, create_graph=True)[0]
        d2udt2 = torch.autograd.grad(dudt.sum(), t_physics, create_graph=True)[0]
        residual = m * d2udt2 + mu * dudt + k * u_phys
        loss_phys = torch.mean(residual**2)

        total_loss = loss_data + lambda_phys * loss_phys
        total_loss.backward()
        optimizer.step()

        if i % 3000 == 0:
            print(f"Iteration {i}, Perte Totale: {total_loss.item():.4e} (Data: {loss_data.item():.4e}, Phys: {loss_phys.item():.4e})")
    print(f"Entrainement termine en {time.time() - start_time:.2f}s")
    return model

# --- 4. Lancement des 4 Experiences ---

# Scenario 1: Regression pure, donnees sans bruit
print("\n--- Scenario 1: Regression Pure (sans bruit) ---")
model1 = train_regression_pure((t_data_train, u_data_train_noiseless))

# Scenario 2: Regression pure, donnees bruitees
print("\n--- Scenario 2: Regression Pure (avec bruit) ---")
model2 = train_regression_pure((t_data_train, u_data_train_noisy))

# Scenario 3: PINN, donnees sans bruit
print("\n--- Scenario 3: PINN (sans bruit) ---")
model3 = train_pinn((t_data_train, u_data_train_noiseless))

# Scenario 4: PINN, donnees bruitees
print("\n--- Scenario 4: PINN (avec bruit) ---")
model4 = train_pinn((t_data_train, u_data_train_noisy))

# --- 5. Visualisation Comparative ---

# Recuperation des predictions pour tous les modeles
u_pred1 = model1(t_full_domain).detach()
u_pred2 = model2(t_full_domain).detach()
u_pred3 = model3(t_full_domain).detach()
u_pred4 = model4(t_full_domain).detach()

fig, axs = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
fig.suptitle("Comparaison de l'Extrapolation : Regression Pure vs. PINN", fontsize=20)

# Donnees pour les graphiques
models_preds = [u_pred1, u_pred2, u_pred3, u_pred4]
titles = [
    "1. Regression Pure (Donnees sans bruit)",
    "2. Regression Pure (Donnees bruitees)",
    "3. PINN (Donnees sans bruit)",
    "4. PINN (Donnees bruitees)"
]
train_datas = [u_data_train_noiseless, u_data_train_noisy, u_data_train_noiseless, u_data_train_noisy]
colors = ['red', 'red', 'green', 'green']

for i, ax in enumerate(axs.flat):
    # Solution exacte
    ax.plot(t_full_domain.cpu().numpy(), u_exact_full.cpu().numpy(), label="Solution Analytique", color="gray", linestyle="--", linewidth=2)
    
    # Prediction du modele
    ax.plot(t_full_domain.cpu().numpy(), models_preds[i].cpu().numpy(), label="Prediction du Modele", color=colors[i], linewidth=2.5)
    
    # Donnees d'entrainement
    ax.scatter(t_data_train.cpu().numpy(), train_datas[i].cpu().numpy(), label="Donnees d'entrainement", color="blue", zorder=5, s=40)
    
    # Zone d'extrapolation
    ax.axvspan(t_train_limit, 1, color='orange', alpha=0.15, label="Zone d'extrapolation")
    
    ax.set_title(titles[i], fontsize=14)
    ax.grid(True, linestyle=':')
    ax.legend()
    ax.set_xlabel("Temps (t)")
    ax.set_ylabel("Deplacement (u)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

