# DBQVM: Dossey–Bohmian Quantum Virtual Machine

This project implements both a **Dossey–Bohmian Quantum Virtual Machine (DBQVM)** and an interactive **Quantum Operating System (QOS)** shell based on a **Bohmian Holographic Simulation Theory**.

Unlike traditional quantum simulators that rely on statistical sampling and collapse postulates, this simulator evolves particles **deterministically** using **Bohmian mechanics** and simulates entangled wave effects in a **real-time holographic field grid**.

The environment allows both **wavefunction-based computation** and **agent-based quantum field interaction**, visualizing guided particle evolution across a simulated Hilbert space.

---

## 🚀 Features

* 2D quantum wavefield simulation with Bohmian agent dynamics
* Deterministic wave-particle duality via the quantum potential
* Field decay, Gaussian emission, and live wavefunction animation
* Programmable wave sources: single slit, double slit, or general emitters
* Support for encoding and decoding text into wavefunctions
* Internal Hilbert space comparisons and overlap checks
* Real-time CLI commands for quantum shell control
* Integrated quantum memory (wavefunction DB)
* Pre-entangled velocity updates based on gradient phase
* Visualization of quantum interference, slits, and trajectories

---

## 📁 Files

* `quantum_wave_os_bohmian_hilbert_boosted.py`: Main simulation and terminal
* `dbqvm_simulation.py`: Original 1D simulation
* `README.md`: This file

---

## 📦 Requirements

* Python 3.8+
* `numpy`
* `matplotlib`

Install dependencies with:

```bash
pip install numpy matplotlib
```

---

## ▶️ How to Run

Launch the quantum OS CLI with:

```bash
python quantum_wave_os_bohmian_hilbert.py
```

---

## ⌨️ Terminal Commands

| Command              | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `emit single`       | Emit wave and align particles toward a single slit          |
| `emit slits`        | Emit dual slits and align particles toward both slits       |
| `emit general`      | Emit general Gaussian wave centered on the field            |
| `setup single`      | Draw single slit on the barrier                             |
| `setup slits`       | Draw dual slit configuration                                |
| `clear slits`       | Remove all slit barriers                                    |
| `encode <text>`     | Encode text into the wavefunction                           |
| `decode [n]`        | Decode recent wavefunction into text (first n chars)       |
| `store <text> :: <tag>` | Store text as a quantum state                         |
| `show`              | List documents and wave states                              |
| `exit`              | Exit the terminal                                            |

---

## 🧠 Theory Summary

This simulator follows the tenets of:

* **Bohmian mechanics** – particles follow real paths, driven by the quantum potential
* **Holographic theory** – all information resides on a multidimensional surface
* **Hilbert space evolution** – wavefunctions evolve via Schrödinger-like rules

### Velocity Equation

Particles follow the Bohmian velocity field:

```
v(x, t) = (ℏ / m) · Im[∇ψ(x,t) / ψ(x,t)]
```

### Encoding

Text can be encoded as perturbations in the wavefunction. You can save and compare quantum states using inner product similarity. This serves as a proof-of-concept quantum memory interface.

---

## 📊 Benchmarks & Tests

Use emitted waves and slit configurations to visualize:

* Interference
* Trajectory distribution
* Hilbert space overlap via `compare_to_reference()`

---

## 🔖 Citations & References

* Bohm, D. (1952). A Suggested Interpretation of the Quantum Theory…
* Holland, P. R. (1995). *The Quantum Theory of Motion*
* Susskind, L. (1995). *The World as a Hologram*
* Valentini, A. (2002). *Subquantum Information and Computation*
* Richards, D. (2024). *Dossey–Bohmian Holographic Simulation Theory* (internal drafts)

---

## 🌌 License

MIT License. Build, adapt, expand.

---

## ✨ Created By

Dossey Richards III — Founder of the Dossey–Bohmian Holographic Simulation Theory
