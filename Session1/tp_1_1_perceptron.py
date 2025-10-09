import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Generate linearly separable data
n = 100
X = np.r_[np.random.randn(n, 2) + np.array([2.0, 2.0]),
          np.random.randn(n, 2) + np.array([-2.0, -2.0])]
y = np.r_[np.ones(n), -np.ones(n)]

# Shuffle data
perm = np.random.permutation(2*n)
X, y = X[perm], y[perm]

# Perceptron algorithm
w = np.zeros(2)
b = 0.0
eta = 0.1
for epoch in range(10):
    errors = 0
    for xi, yi in zip(X, y):
        if yi * (np.dot(w, xi) + b) <= 0:
            w += eta * yi * xi
            b += eta * yi
            errors += 1
    print(f"Epoch {epoch}, errors: {errors}")
    if errors == 0:
        print("Convergence reached.")
        break

# Visualization
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
Z = (w[0]*xx + w[1]*yy + b)
plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap=plt.cm.bwr)
plt.scatter(X[:, 0], X[:, 1], c=(y > 0), cmap='bwr', edgecolor='k')
plt.title("Frontiere de decision du Perceptron")
plt.axis('equal');
plt.show()

