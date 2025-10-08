# -*- coding: utf-8 -*-
"""
TP4 : Resolution d'une EDO par PINN (Probleme Direct)

Objectif :
- Implementer un PINN pour simuler la trajectoire de l'oscillateur harmonique
  amorti sans aucune donnee de trajectoire, en utilisant uniquement l'EDO
  et les conditions initiales.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration du probleme physique et du modele ---
# Parametres physiques de l'oscillateur (connus dans ce scenario)
m, mu, k = 1.0, 2.0, 100.0  # masse, coef. frottement, raideur ressort
d, w0 = mu / (2 * m), np.sqrt(k / m)

# Architecture du reseau de neurones
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

# --- 2. Definition de la fonction de perte du PINN ---
def pinn_loss(model, t_physics, t_boundary, lambda1=1e-1, lambda2=1e-4):
    # Perte aux conditions initiales
    u_bc = model(t_boundary)
    dudt_bc = torch.autograd.grad(u_bc, t_boundary, torch.ones_like(u_bc), create_graph=True)[0]
    
    loss_bc1 = (u_bc - 1.0)**2  # u(0) = 1
    loss_bc2 = dudt_bc**2        # u'(0) = 0
    
    # Perte physique (residu de l'EDO)
    u_p = model(t_physics)
    dudt_p = torch.autograd.grad(u_p, t_physics, torch.ones_like(u_p), create_graph=True)[0]
    d2udt2_p = torch.autograd.grad(dudt_p, t_physics, torch.ones_like(dudt_p), create_graph=True)[0]
    
    residual = m * d2udt2_p + mu * dudt_p + k * u_p
    loss_physics = torch.mean(residual**2)
    
    # Perte totale
    loss = loss_bc1 + lambda1 * loss_bc2 + lambda2 * loss_physics
    return loss.squeeze()

# --- 3. Boucle d'entrainement ---
pinn = FCN(1, 1, 32, 3)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# Points de collocation pour la perte physique
t_physics = torch.linspace(0, 1, 30, requires_grad=True).view(-1, 1)
# Point pour les conditions initiales
t_boundary = torch.tensor([[0.]], requires_grad=True)

print("Debut de l'entrainement du PINN pour le probleme direct...")
for i in range(15001):
    optimizer.zero_grad()
    loss = pinn_loss(pinn, t_physics, t_boundary)
    loss.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        print(f"Iteration {i}, Perte: {loss.item():.4e}")
print("Entrainement termine.")

# --- 4. Visualisation des resultats ---
# Solution analytique pour comparaison
def exact_solution(d, w0, t):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)
    return exp * 2 * A * cos

t_test = torch.linspace(0, 1, 300).view(-1, 1)
u_exact = exact_solution(d, w0, t_test)
u_pinn = pinn(t_test).detach()

plt.figure(figsize=(10, 6))
plt.plot(t_test.numpy(), u_pinn.numpy(), label="Solution PINN", color="red")
plt.plot(t_test.numpy(), u_exact.numpy(), label="Solution Analytique", color="gray", linestyle="--")

plt.scatter(t_physics.detach().numpy(), np.zeros_like(t_physics.detach().numpy()), 
            marker='x', color='blue', label='Points de Collocation')
plt.title("Probleme Direct : Resolution d'EDO par PINN")
plt.xlabel("Temps (t)")
plt.ylabel("Deplacement (u)")
plt.legend()
plt.grid(True)
plt.show()


