# -*- coding: utf-8 -*-
"""
TP2 : Exploration du compromis Biais-Variance avec un MLP

Objectifs :
1.  Visualiser le sous-apprentissage (biais eleve) avec un modele trop simple.
2.  Visualiser le sur-apprentissage (variance elevee) avec un modele trop complexe.
3.  Observer les courbes d'apprentissage typiques du sur-apprentissage.
4.  Mettre en uvre et observer l'effet regularisateur du dropout.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Preparation des donnees ---

# Generation du jeu de donnees "moons"
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Diviser en ensembles d'entrainement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardisation des donnees
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Conversion en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# --- 2. Definition du Modele MLP ---

class MLP(nn.Module):
    def __init__(self, hidden_layers, dropout_p=0.0):
        super(MLP, self).__init__()
        layers = []
        input_dim = 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 2)) # 2 classes de sortie
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- 3. Fonctions d'entrainement et de visualisation ---

def train_and_evaluate(model, X_train, y_train, X_val, y_val, epochs=5000, lr=0.005):
    """Entraine le modele et retourne l'historique des pertes."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluation a chaque 10 epoques
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, predicted = torch.max(val_outputs.data, 1)
                accuracy = (predicted == y_val).sum().item() / len(y_val)
                
                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss.item())
                history['val_acc'].append(accuracy)

    return history

def plot_decision_boundary(model, X, y, title):
    """Affiche la frontiere de decision du modele."""
    model.eval()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    with torch.no_grad():
        Z = model(grid_tensor)
        _, Z = torch.max(Z, 1)
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z.numpy(), cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title(title)
    plt.xlabel("Caracteristique 1")
    plt.ylabel("Caracteristique 2")

def plot_learning_curves(history, title):
    """Affiche les courbes d'apprentissage (perte)."""
    plt.plot(history['train_loss'], label='Perte d\'entrainement')
    plt.plot(history['val_loss'], label='Perte de validation')
    plt.title(title)
    plt.xlabel("Iterations (x10 epoques)")
    plt.ylabel("Perte (Entropie Croisee)")
    plt.legend()
    plt.ylim(0, 1) # Limiter l'axe y pour une meilleure lisibilite

# --- 4. Experimentations ---

plt.figure(figsize=(18, 12))
plt.suptitle("Exploration du Compromis Biais-Variance et de la Regularisation", fontsize=16)

# Experience 1: Sous-apprentissage (modele trop simple)
underfit_model = MLP(hidden_layers=[8])
history_underfit = train_and_evaluate(underfit_model, X_train, y_train, X_val, y_val)
plt.subplot(2, 3, 1)
plot_decision_boundary(underfit_model, X_train.numpy(), y_train.numpy(), "1a. Sous-apprentissage - Frontiere")
plt.subplot(2, 3, 4)
plot_learning_curves(history_underfit, "1b. Sous-apprentissage - Courbes")

# Experience 2: Sur-apprentissage (modele trop complexe)
overfit_model = MLP(hidden_layers=[256, 256, 256])
history_overfit = train_and_evaluate(overfit_model, X_train, y_train, X_val, y_val)
plt.subplot(2, 3, 2)
plot_decision_boundary(overfit_model, X_train.numpy(), y_train.numpy(), "2a. Sur-apprentissage - Frontiere")
plt.subplot(2, 3, 5)
plot_learning_curves(history_overfit, "2b. Sur-apprentissage - Courbes")
print(f"Accuracy finale sur-apprentissage: {history_overfit['val_acc'][-1]:.3f}")


# Experience 3: Modele regularise avec Dropout
regularized_model = MLP(hidden_layers=[256, 256, 256], dropout_p=0.5)
history_regularized = train_and_evaluate(regularized_model, X_train, y_train, X_val, y_val)
plt.subplot(2, 3, 3)
plot_decision_boundary(regularized_model, X_train.numpy(), y_train.numpy(), "3a. Regularisation (Dropout) - Frontiere")
plt.subplot(2, 3, 6)
plot_learning_curves(history_regularized, "3b. Regularisation (Dropout) - Courbes")
print(f"Accuracy finale avec dropout: {history_regularized['val_acc'][-1]:.3f}")


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

