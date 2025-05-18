# quantum_os.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
import threading
import time
import platform

# --- Config ---
GRID_SIZE = (64, 64)
NUM_AGENTS = 5
STEPS = 500
DT = 0.2
NOISE_STD = 0.1
FIELD_DECAY = 0.95
VERSION = "v0.9-beta"
ENTANGLE_STRENGTH = 0.5  # Influence factor from entangled agents

# --- State ---
np.random.seed(42)
field = np.zeros(GRID_SIZE)
agents = [
    {
        "pos": np.random.uniform(0, GRID_SIZE[0], size=2),
        "id": i,
        "memory": [],
        "entangled_with": []  # New: Track which other agents this one is entangled with
    }
    for i in range(NUM_AGENTS)
]
start_time = time.time()

# --- Field API ---
def emit_field(x, y, amplitude):
    xi, yi = int(x), int(y)
    if 0 <= xi < GRID_SIZE[0] and 0 <= yi < GRID_SIZE[1]:
        field[xi, yi] += amplitude

def kill_agent(aid):
    agents = [a for a in agents if a['id'] != aid]
    print(f"💀 Killed agent {aid}")

# --- Extended: Emit to entangled partners too ---
def emit_with_entanglement(agent, amplitude):
    emit_field(*agent["pos"], amplitude)
    for eid in agent["entangled_with"]:
        partners = [a for a in agents if a["id"] == eid]
        for partner in partners:
            emit_field(*partner["pos"], amplitude * ENTANGLE_STRENGTH)

# --- Gradient with entanglement influence ---
def sense_gradient(pos):
    x, y = pos
    dx = 1e-2
    fx = field[int(np.clip(x + dx, 0, GRID_SIZE[0]-1)) % GRID_SIZE[0], int(y)]
    fy = field[int(x), int(np.clip(y + dx, 0, GRID_SIZE[1]-1)) % GRID_SIZE[1]]
    f0 = field[int(x), int(y)]
    grad_x = (fx - f0) / dx
    grad_y = (fy - f0) / dx
    return np.array([grad_x, grad_y])

def blended_gradient(agent):
    base_grad = sense_gradient(agent["pos"])
    # Sum gradients of entangled partners
    total_grad = base_grad.copy()
    for eid in agent["entangled_with"]:
        partners = [a for a in agents if a["id"] == eid]
        for partner in partners:
            total_grad += sense_gradient(partner["pos"]) * ENTANGLE_STRENGTH
    return total_grad

# --- Kernel ---
def step_agents():
    for agent in agents:
        pos = agent["pos"]
        gradient = blended_gradient(agent)
        noise = np.random.normal(0, NOISE_STD, size=2)
        velocity = -gradient + noise
        pos += velocity * DT
        pos = np.clip(pos, [0, 0], np.array(GRID_SIZE) - 1)
        agent["pos"] = pos
        agent["memory"].append(pos.copy())
        emit_with_entanglement(agent, amplitude=0.2)

def decay_field():
    global field
    field *= FIELD_DECAY

# --- State Management ---
def save_state(filename="quantum_state.qos"):
    with open(filename, "wb") as f:
        pickle.dump({"field": field, "agents": agents}, f)
    print(f"✅ State saved to {filename}")

def load_state(filename="quantum_state.qos"):
    global field, agents
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            state = pickle.load(f)
            field = state["field"]
            agents = state["agents"]
        print(f"📥 State loaded from {filename}")
    else:
        print("❌ State file not found.")

# --- Code Execution API ---
def run_user_program(code: str):
    allowed_globals = {
        "emit_field": emit_field,
        "sense_gradient": sense_gradient,
        "agents": agents,
        "field": field,
        "np": np
    }
    try:
        exec(code, allowed_globals)
        print("✅ Code executed.")
    except Exception as e:
        print(f"❌ Error: {e}")

# --- Terminal Thread ---
def terminal():
    global agents
    uptime = lambda: time.time() - start_time
    banner = f"""
╔═════════════════════════════════════════════════════════╗
║            DOSSEY-BOHM HQPU OPERATING SYSTEM            ║
║         Holographic Quantum Processing Unit (HQPU)     ║
╚═════════════════════════════════════════════════════════╝
Inspired by the causal realism of David Bohm—
whose contributions to quantum theory remain underappreciated.

System Info:
  Version: {VERSION}
  Platform: {platform.system()} {platform.release()}
  Python: {platform.python_version()}
  Agents Loaded: {len(agents)}
  Grid Size: {GRID_SIZE[0]}x{GRID_SIZE[1]}
  Runtime Uptime: {uptime():.2f} sec (auto updates below)
"""
    print(banner)
    while True:
        cmd = input("QOS > ").strip().lower()
        if cmd == "exit":
            print("🛑 Exiting terminal.")
            break
        elif cmd.startswith("emit"):
            try:
                _, x, y, amp = cmd.split()
                emit_field(float(x), float(y), float(amp))
                print(f"🔸 Emitted field at ({x}, {y}) with amp {amp}")
            except:
                print("❌ Usage: emit <x> <y> <amplitude>")
        elif cmd == "list agents":
            for a in agents:
                print(f"Agent {a['id']} at {a['pos']}, entangled_with: {a['entangled_with']}")
        elif cmd.startswith("entangle"):
            try:
                _, a1, a2 = cmd.split()
                a1, a2 = int(a1), int(a2)
                for a in agents:
                    if a["id"] == a1:
                        a["entangled_with"].append(a2)
                    if a["id"] == a2:
                        a["entangled_with"].append(a1)
                print(f"🔗 Entangled agent {a1} <--> agent {a2}")
            except:
                print("❌ Usage: entangle <id1> <id2>")
        elif cmd.startswith("save"):
            _, name = cmd.split() if len(cmd.split()) > 1 else ("save", "quantum_state.qos")
            save_state(name)
        elif cmd.startswith("load"):
            _, name = cmd.split() if len(cmd.split()) > 1 else ("load", "quantum_state.qos")
            load_state(name)
        elif cmd.startswith("kill"):
            parts = cmd.split()
            if len(parts) != 2:
                print("❌ Usage: kill <agent_id>")
                continue
            try:
                aid = int(parts[1])
                kill_agent(aid)
            except ValueError:
                print("❌ Invalid ID")
                continue
            agents = [a for a in agents if a["id"] != aid]
            print(f"💀 Killed agent {aid}")
        elif cmd.startswith("spawn"):
            try:
                _, x, y = cmd.split()
                new_id = max([a["id"] for a in agents]) + 1 if agents else 0
                agents.append({"pos": np.array([float(x), float(y)]), "id": new_id, "memory": [], "entangled_with": []})
                print(f"✨ Spawned agent {new_id} at ({x}, {y})")
            except:
                print("❌ Usage: spawn <x> <y>")
        elif cmd == "run":
            print("Enter Python code. End with an empty line:")
            lines = []
            while True:
                line = input(">>> ")
                if line.strip() == "":
                    break
                lines.append(line)
            code_block = "\n".join(lines)
            run_user_program(code_block)
        elif cmd == "help":
            print("Commands: emit <x> <y> <amp>, list agents, entangle <id1> <id2>, save <file>, load <file>, kill <id>, spawn <x> <y>, run, exit")
        else:
            print("❓ Unknown command. Type 'help'.")

# --- Visualization ---
def animate(i):
    decay_field()
    step_agents()
    ax.clear()
    ax.imshow(field.T, cmap='inferno', origin='lower', vmin=0, vmax=1)
    for agent in agents:
        x, y = agent["pos"]
        ax.plot(x, y, 'go')
    ax.set_title(f"Quantum OS Tick {i}")

# --- Launch ---
threading.Thread(target=terminal, daemon=True).start()
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=50, repeat=False)
plt.show()