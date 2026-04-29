import numpy as np
import matplotlib.pyplot as plt

# 1. Unmanned Aerial Vehicle (UAV) and Defensive Position Distance Relationship
# Parameters for the defensive position
R_j = 100  # Threat radius (meters)
x_defense, y_defense = 0, 0  # Defense position coordinates

# UAV positions around the defense position
theta = np.linspace(0, 2 * np.pi, 100)
x_uav = R_j * np.cos(theta)
y_uav = R_j * np.sin(theta)

# Plotting UAV and defensive position
plt.figure(figsize=(8, 8))
plt.plot(x_uav, y_uav, label="Threat Boundary ($R_j$)", linestyle='--', color='red')
plt.scatter(x_defense, y_defense, color='blue', label="Defensive Position", s=100)
plt.scatter(50, 50, color='green', label="UAV Position ($d_{ij}$)", s=100)
plt.arrow(0, 0, 50, 50, head_width=5, head_length=10, fc='black', ec='black', label='$d_{ij}$')
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("UAV-Defensive Position Distance Relationship")
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()

# 2. Probability Function Curve for Threat (P_ij vs. d_ij)
# Distance range (0 to 2R_j)
d_ij = np.linspace(1, 2 * R_j, 500)
P_ij = np.where(d_ij <= R_j, 1 - np.exp(-R_j / (d_ij ** 2)), 0)

# Plotting the probability function
plt.figure(figsize=(8, 6))
plt.plot(d_ij, P_ij, color='purple', label="$P_{ij}$ Curve")
plt.axvline(R_j, color='red', linestyle='--', label="Threat Radius ($R_j$)")
plt.xlabel("Distance ($d_{ij}$ in meters)")
plt.ylabel("Threat Probability ($P_{ij}$)")
plt.title("Threat Probability Function ($P_{ij}$ vs. $d_{ij}$)")
plt.grid()
plt.legend()
plt.show()
