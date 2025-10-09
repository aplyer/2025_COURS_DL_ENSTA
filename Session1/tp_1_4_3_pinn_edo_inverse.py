# -*- coding: utf-8 -*-
"""
TP5 : Estimation de parametre par PINN (Probleme Inverse)

Objectif :
- Utiliser un PINN pour estimer le coefficient de frottement inconnu (mu)
  d'un oscillateur harmonique a partir de quelques mesures bruitees.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration du probleme et generation des donnees ---
# Vrais parametres physiques (mu est celui que nous voulons decouvrir)
m_true, mu_true, k_true = 1.0, 4.0, 100.0
d_true, w0_true = mu_true / (2 * m_true), np.sqrt(k_true / m_true)

# Solution analytique pour generer les donnees
def exact_solution(d, w0, t):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)
    return exp * 2 * A * cos

# Generation de donnees d'observation bruitees
N_obs = 40
noise_level = 0.05
t_obs = torch.rand(N_obs).view(-1, 1) * 0.7  # Observations sur une partie du domaine
u_obs = exact_solution(d_true, w0_true, t_obs) + noise_level * torch.randn_like(t_obs)

# Architecture du reseau
class FCN(nn.Module):
    "Fully-Connected Network"
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

# --- 2. Entrainement du PINN pour le probleme inverse ---
pinn_inverse = FCN(1, 1, 32, 3)
# mu est maintenant un parametre apprenable !
# Initialisation arbitraire
log_mu = torch.nn.Parameter(torch.log(torch.tensor([1.0])), requires_grad=True)

# L'optimiseur inclut les poids du reseau ET le parametre mu
optimizer = torch.optim.Adam(list(pinn_inverse.parameters()) + [log_mu], lr=1e-3)
criterion = nn.MSELoss()

# Points de collocation pour la perte physique
t_physics = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)

lambda_physics = 1e-2  # Poids pour le terme physique
mu_history = []

print("Debut de l'entrainement du PINN pour le probleme inverse...")
for i in range(20001):
    optimizer.zero_grad()
    
    # On utilise exp(log_mu) pour garantir que mu reste positif
    mu = torch.exp(log_mu)
    mu_history.append(mu.item())
    
    # Perte sur les donnees
    u_pred_obs = pinn_inverse(t_obs)
    loss_data = criterion(u_pred_obs, u_obs)
    
    # Perte physique
    u_p = pinn_inverse(t_physics)
    dudt_p = torch.autograd.grad(u_p, t_physics, torch.ones_like(u_p), create_graph=True)[0]
    d2udt2_p = torch.autograd.grad(dudt_p, t_physics, torch.ones_like(dudt_p), create_graph=True)[0]
    
    residual = m_true * d2udt2_p + mu * dudt_p + k_true * u_p
    loss_physics = torch.mean(residual**2)
    
    # Perte totale
    loss = loss_data + lambda_physics * loss_physics
    
    loss.backward()
    optimizer.step()
    
    if i % 2000 == 0:
        print(f"Iteration {i}, Perte: {loss.item():.4e}, mu estime: {mu.item():.4f}")
print("Entrainement termine.")

# --- 3. Visualisation des resultats ---
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1: Comparaison des trajectoires
t_test = torch.linspace(0, 1, 300).view(-1, 1)
u_exact = exact_solution(d_true, w0_true, t_test)
u_pinn_inverse = pinn_inverse(t_test).detach()

axs[0].plot(t_test.numpy(), u_exact.numpy(), label="Solution Analytique (Vrai mu)", color="gray", linestyle="--", linewidth=2)
axs[0].plot(t_test.numpy(), u_pinn_inverse.numpy(), label="Solution PINN (mu estime)", color="red")
axs[0].scatter(t_obs.numpy(), u_obs.numpy(), color='blue', label='Donnees observees (bruitees)', marker='x')
axs[0].set_title("Probleme Inverse : Reconstruction de la Trajectoire")
axs[0].set_xlabel("Temps (t)")
axs[0].set_ylabel("Deplacement (u)")
axs[0].legend()
axs[0].grid(True)

# Graphique 2: Convergence du parametre mu
axs[1].plot(mu_history, color='green', label='mu estime')
axs[1].axhline(y=mu_true, color='black', linestyle='--', label=f'Vraie valeur de mu ({mu_true})')
axs[1].set_title("Convergence du Parametre Inconnu (mu)")
axs[1].set_xlabel("Iterations d'entrainement")
axs[1].set_ylabel("Valeur de mu")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

print(f"\nValeur reelle de mu: {mu_true}")
print(f"Valeur finale de mu estimee par le PINN: {torch.exp(log_mu).item():.4f}")


