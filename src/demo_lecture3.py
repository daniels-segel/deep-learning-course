import numpy as np
import matplotlib.pyplot as plt

# demo of optimization

# ==========================================
# 1. Construct a Complicated 2D Function
# ==========================================
# A complex surface featuring multiple peaks, valleys, and saddle points
def f(x, y):
    return (
        np.sin(x) * np.exp((1 - np.cos(y))**2) +
        np.cos(y) * np.exp((1 - np.sin(x))**2) +
        (x - y)**2 / 10.0
    )

# Partial derivatives for gradient computation
def grad_f(x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

# ==========================================
# 2. Optimization Parameters & Initial Point
# ==========================================
start_point = np.array([-0.5, 0])  # Step 2: Initial point
learning_rate = 0.002               # Step size factor
iterations = 4000

# Track the history of points for visualization
path = [start_point.copy()]
current_point = start_point.copy()

# ==========================================
# 3 & 4. Calculate Gradient and Take Step
# ==========================================
for step in range(iterations):
    grad = grad_f(current_point[0], current_point[1])  # Step 3: Calculate gradient
    current_point -= learning_rate * grad
    #print(f'{grad}')              # Step 4: Step in opposite gradient direction
    path.append(current_point.copy())

path = np.array(path)

# ==========================================
# 5. Show / Visualization
# ==========================================
x_range = np.linspace(-3, 3, 200)
y_range = np.linspace(-1, 4, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

fig = plt.figure(figsize=(14, 6))

# Subplot 1: 3D Surface Plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75, edgecolor='none')
z_path = f(path[:, 0], path[:, 1])
ax1.plot(path[:, 0], path[:, 1], z_path, color='red', marker='o', markersize=4, linewidth=2, label='GD Path')
ax1.set_title("Gradient Descent on 3D Surface")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Loss f(X, Y)")
ax1.legend()

# Subplot 2: 2D Contour Map
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
fig.colorbar(contour, ax=ax2, label='Loss Value')
ax2.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=4, linewidth=2, label='GD Steps')
ax2.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start Point')
ax2.plot(path[-1, 0], path[-1, 1], 'rx', markersize=10, markeredgewidth=3, label='Final Point')
ax2.set_title("2D Contour View of Gradient Steps")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()

plt.tight_layout()
plt.show()