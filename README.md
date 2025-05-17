
# BQVM: Dossey–Bohmian Quantum Virtual Machine

This project implements a minimal **Bohmian Quantum Virtual Machine (BQVM)** using the **Dossey–Bohmian Holographic Simulation Theory**.

Unlike traditional quantum simulators which rely on statistical sampling and collapse postulates, this simulator evolves particles **deterministically** along paths guided by the quantum wavefunction using **Bohmian mechanics**.

---

## 🚀 Features

- 1D Gaussian wave packet simulation
- Bohmian guidance equation for particle velocity
- Deterministic particle trajectory evolution
- Visualization of wave-guided paths

---

## 📁 Files

- `bqvm_simulation.py`: Main simulation script (run this!)
- `README.md`: This file

---

## 📦 Requirements

- Python 3.7+
- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install numpy matplotlib
```

---

## 🧠 Theory Summary

This simulator is based on the principle that:
- Quantum systems evolve deterministically under the Schrödinger equation.
- Particles follow real paths dictated by a velocity field derived from the wavefunction:

 v(x, t) = (ħ / m) · Im[∇ψ(x,t) / ψ(x,t)]

- There is no collapse—just dynamic guidance through a holographic quantum field.

---

## ▶️ How to Run

```bash
python bqvm_simulation.py
```

This will produce a plot showing the deterministic trajectories of particles under a Gaussian quantum wave packet.

---

## 🌌 License

MIT License. Feel free to build, extend, and repurpose. Attribution appreciated.

---

## ✨ Created By

Dossey Richards — founder of the Dossey–Bohmian Holographic Simulation Theory.
