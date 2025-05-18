# DBQVM: Dossey–Bohmian Quantum Virtual Machine

This project implements both a **Dossey–Bohmian Quantum Virtual Machine (DBQVM)** and an interactive **Quantum Operating System (QOS)** shell based on the **Dossey–Bohmian Holographic Simulation Theory**.

Unlike traditional quantum simulators which rely on statistical sampling and collapse postulates, this simulator evolves particles **deterministically** along paths guided by the quantum wavefunction using **Bohmian mechanics**.

The simulation represents a quantum field system where agents move, interact, entangle, and influence a shared holographic grid via deterministic evolution rules guided by quantum-like gradients.

---

## 🚀 Features

* Quantum grid simulation with entangled agent behavior
* Deterministic field-guided agent motion
* Real-time field emission and decay
* Agent memory, spawning, entanglement, and field manipulation
* Interactive terminal with custom command set
* Real-time matplotlib animation
* 1D Gaussian wave packet simulation (via `dbqvm_simulation.py`)
* Bohmian guidance equation for particle velocity
* Deterministic particle trajectory visualization

---

## 📁 Files

* `quantum_os_exec.py`: Main executable Quantum OS simulation
* `dbqvm_simulation.py`: Original Dossey–Bohmian 1D simulation
* `README.md`: This file

---

## 📆 Requirements

* Python 3.7+
* `numpy`
* `matplotlib`

Install dependencies with:

```bash
pip install numpy matplotlib
```

---

## ▶️ How to Run

Launch the quantum OS with:

```bash
python quantum_os_exec.py
```

To run the 1D Gaussian wave packet simulation:

```bash
python dbqvm_simulation.py
```

This will produce a plot showing the deterministic trajectories of particles under a Gaussian quantum wave packet.

---

## ⌨️ Terminal Commands

Inside the terminal, use the following commands:

| Command            | Example           | Description                                               |
| ------------------ | ----------------- | --------------------------------------------------------- |
| `emit x y amp`     | `emit 10 10 0.5`  | Emit energy to the field at (x, y) with given amplitude   |
| `list agents`      |                   | List all agents and their positions/entanglements         |
| `entangle id1 id2` | `entangle 0 2`    | Entangle two agents to share field influence              |
| `kill id`          | `kill 3`          | Remove agent with specified ID                            |
| `spawn x y`        | `spawn 25 32`     | Spawn a new agent at the given coordinates                |
| `save [filename]`  | `save state1.qos` | Save current field and agent state to a file              |
| `load [filename]`  | `load state1.qos` | Load a saved field/agent state                            |
| `run`              |                   | Enter Python code interactively to modify field or agents |
| `help`             |                   | Show available commands                                   |
| `exit`             |                   | Exit the terminal                                         |

---

## 🧠 Theory Summary

This simulator is based on the principle that:

* Quantum systems evolve deterministically under the Schrödinger equation.
* Particles follow real paths dictated by a velocity field derived from the wavefunction:

`v(x, t) = (ℏ / m) · Im[∇ψ(x,t) / ψ(x,t)]`

* There is no collapse—just dynamic guidance through a holographic quantum field.

This system blends:

* **Bohmian mechanics** (deterministic paths guided by quantum potentials)
* **Holographic simulation concepts**
* **Field interaction via real-valued gradients and entanglement influence**

Agents evolve deterministically:

* Sense gradients of the scalar field
* Are influenced by entangled partners
* Emit back into the field

The system forms a feedback loop resembling a quantum-classical hybrid.

---

## 🔖 Citations & References

* Bohm, D. (1952). A Suggested Interpretation of the Quantum Theory in Terms of "Hidden" Variables I & II. *Physical Review*, 85(2), 166–193.
* Holland, P. R. (1995). *The Quantum Theory of Motion: An Account of the de Broglie-Bohm Causal Interpretation of Quantum Mechanics*. Cambridge University Press.
* Susskind, L. (1995). The World as a Hologram. *Journal of Mathematical Physics*, 36(11), 6377–6396.
* Valentini, A. (2002). Subquantum Information and Computation. *Pramana*, 59(2), 269–277.
* 'Dossey–Bohmian Holographic Simulation Theory', internal drafts, D. Richards (2024).

---

## 🌌 License

MIT License. Build, repurpose, simulate.

---

## ✨ Created By

Dossey Richards III — Founder of the Dossey–Bohmian Holographic Simulation Theory
