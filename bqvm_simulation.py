
import numpy as np
import matplotlib.pyplot as plt

# ---- Wavefunction Engine ----
class Wavefunction1D:
    def __init__(self, x, x0=0.0, k0=5.0, sigma=0.2, mass=1.0, hbar=1.0):
        self.x = x
        self.x0 = x0
        self.k0 = k0
        self.sigma = sigma
        self.mass = mass
        self.hbar = hbar

    def gaussian_packet(self, t=0.0):
        prefactor = 1.0 / (np.sqrt(self.sigma * np.sqrt(np.pi)))
        phase = np.exp(1j * (self.k0 * self.x - (self.hbar * self.k0**2 / (2 * self.mass)) * t))
        envelope = np.exp(-(self.x - self.x0 - (self.hbar * self.k0 / self.mass) * t)**2 / (2 * self.sigma**2))
        return prefactor * envelope * phase

# ---- Bohmian Guidance Engine ----
class BohmianEngine:
    def __init__(self, psi_func, x_grid, hbar=1.0, mass=1.0):
        self.psi_func = psi_func
        self.x_grid = x_grid
        self.hbar = hbar
        self.mass = mass

    def velocity_field(self, x, t):
        dx = 1e-5
        psi_val = self.psi_func(x, t)
        psi_plus = self.psi_func(x + dx, t)
        dpsi_dx = (psi_plus - psi_val) / dx
        return (self.hbar / self.mass) * np.imag(dpsi_dx / psi_val)

    def evolve_trajectory(self, x0, t_array):
        x_traj = np.zeros_like(t_array)
        x_traj[0] = x0
        for i in range(1, len(t_array)):
            dt = t_array[i] - t_array[i - 1]
            v = self.velocity_field(x_traj[i - 1], t_array[i - 1])
            x_traj[i] = x_traj[i - 1] + v * dt
        return x_traj

# ---- Main Simulation Runner ----
x_grid = np.linspace(-5, 5, 1000)
t_grid = np.linspace(0, 1.0, 200)
wave = Wavefunction1D(x_grid)

# Interpolation method for ψ(x, t)
def psi_interp(x, t):
    psi_full = wave.gaussian_packet(t)
    index = np.abs(x_grid - x).argmin()
    return psi_full[index]

engine = BohmianEngine(psi_func=psi_interp, x_grid=x_grid)

# Simulate trajectories
initial_positions = np.linspace(-2.0, 2.0, 5)
trajectories = [engine.evolve_trajectory(x0, t_grid) for x0 in initial_positions]

# Plot the trajectories
fig, ax = plt.subplots(figsize=(10, 6))
for i, traj in enumerate(trajectories):
    ax.plot(t_grid, traj, label=f"x₀ = {initial_positions[i]:.2f}")
ax.set_title("BQVM Trajectories: Dossey–Bohmian Holographic Simulation")
ax.set_xlabel("Time")
ax.set_ylabel("Particle Position")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
