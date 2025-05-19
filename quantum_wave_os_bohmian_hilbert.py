import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import uuid
import json

# --- Config ---
GRID_SIZE = (64, 64)
NUM_AGENTS = 500
DT = 0.2
NOISE_STD = 0.1
FIELD_DECAY = 0.99
ħ = 1.0
m = 1.0

# --- State ---
np.random.seed(42)
field = np.zeros(GRID_SIZE)
psi_real = np.zeros(GRID_SIZE)
psi_imag = np.zeros(GRID_SIZE)
quantum_db = {}
last_encoded_text = ""
tick = 0
agents = [{"pos": np.random.uniform(0, GRID_SIZE[0], 2), "id": i, "memory": []} for i in range(NUM_AGENTS)]
barrier_mask = np.zeros(GRID_SIZE, bool)

# --- Slit Setup ---
def clear_barrier():
    barrier_mask[:] = False

def setup_single_slit():
    clear_barrier()
    x = GRID_SIZE[0] // 2
    y = GRID_SIZE[1] // 2
    barrier_mask[x, :] = True
    barrier_mask[x, y - 3:y + 3] = False

def setup_double_slit():
    clear_barrier()
    x = GRID_SIZE[0] // 2
    barrier_mask[x, :] = True
    mid = GRID_SIZE[1] // 2
    barrier_mask[x, mid - 10:mid - 4] = False
    barrier_mask[x, mid + 4:mid + 10] = False

# --- Emission ---
def emit_single_slit_wave():
    cx, cy = GRID_SIZE[0] // 4, GRID_SIZE[1] // 2
    for x in range(cx - 5, cx + 5):
        for y in range(cy - 3, cy + 3):
            g = 5 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 5.0)
            psi_real[x, y] += g

def emit_two_slit_wave():
    cx = GRID_SIZE[0] // 4
    for offset in (-10, 10):
        cy = GRID_SIZE[1] // 2 + offset
        for x in range(cx - 5, cx + 5):
            for y in range(cy - 3, cy + 3):
                g = 5 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 5.0)
                psi_real[x, y] += g

# --- Wave Evolution ---
def evolve_wavefunction(dt):
    global psi_real, psi_imag
    lap_r = (np.roll(psi_real, 1, 0) + np.roll(psi_real, -1, 0) + np.roll(psi_real, 1, 1) + np.roll(psi_real, -1, 1) - 4 * psi_real)
    lap_i = (np.roll(psi_imag, 1, 0) + np.roll(psi_imag, -1, 0) + np.roll(psi_imag, 1, 1) + np.roll(psi_imag, -1, 1) - 4 * psi_imag)
    psi_real += (-ħ / (2 * m)) * lap_i * dt
    psi_imag += ( ħ / (2 * m)) * lap_r * dt
    psi_real[barrier_mask] = 0
    psi_imag[barrier_mask] = 0
    norm = np.sqrt(np.sum(psi_real ** 2 + psi_imag ** 2))
    if norm > 0:
        psi_real /= norm
        psi_imag /= norm


# --- Bohmian Mechanics Phase Field and Agent Update ---
def compute_phase_field():
    psi = psi_real + 1j * psi_imag
    phase = np.angle(psi)
    return phase

def update_agents():
    global agents
    phase = compute_phase_field()
    grad_y, grad_x = np.gradient(phase)
    for agent in agents:
        x, y = agent["pos"]
        xi, yi = int(x), int(y)
        if 1 <= xi < GRID_SIZE[0] - 1 and 1 <= yi < GRID_SIZE[1] - 1:
            vx = (ħ / m) * grad_x[xi, yi]
            vy = (ħ / m) * grad_y[xi, yi]
            new_x = x + vx * DT
            new_y = y + vy * DT
            # Reflect at barriers
            if not barrier_mask[int(new_x) % GRID_SIZE[0], int(new_y) % GRID_SIZE[1]]:
                agent["pos"] = np.array([new_x, new_y])
# --- Bohmian Velocity ---
def compute_bohmian_velocity(pos):
    x, y = pos.astype(int)
    ψ = complex(psi_real[x % GRID_SIZE[0], y % GRID_SIZE[1]], psi_imag[x % GRID_SIZE[0], y % GRID_SIZE[1]])
    if abs(ψ) < 1e-6:
        return np.zeros(2)
    grad_x = complex(psi_real[(x + 1) % GRID_SIZE[0], y] - psi_real[(x - 1) % GRID_SIZE[0], y],
                     psi_imag[(x + 1) % GRID_SIZE[0], y] - psi_imag[(x - 1) % GRID_SIZE[0], y]) / 2
    grad_y = complex(psi_real[x, (y + 1) % GRID_SIZE[1]] - psi_real[x, (y - 1) % GRID_SIZE[1]],
                     psi_imag[x, (y + 1) % GRID_SIZE[1]] - psi_imag[x, (y - 1) % GRID_SIZE[1]]) / 2
    return np.array([(ħ / m) * np.imag(grad_x / ψ), (ħ / m) * np.imag(grad_y / ψ)])

# --- Agents ---
def emit_field(x, y, amp):
    xi, yi = int(x), int(y)
    if 0 <= xi < GRID_SIZE[0] and 0 <= yi < GRID_SIZE[1]:
        field[xi, yi] += amp

def step_agents():
    for a in agents:
        if 'vel' in a:
            a['pos'] += a['vel'] * 0.5
        a["pos"] += (compute_bohmian_velocity(a["pos"]) + np.random.normal(0, NOISE_STD, 2)) * DT
        a["pos"] = np.clip(a["pos"], [0, 0], np.array(GRID_SIZE) - 1)
        a["memory"].append(a["pos"].copy())
        emit_field(*a["pos"], 0.2)

def decay_field():
    global field
    field *= FIELD_DECAY

# --- Encoding ---
def encode_text_into_wavefunction(text):
    global last_encoded_text
    last_encoded_text = text
    wave = np.zeros(GRID_SIZE, dtype=np.complex128)
    x = np.arange(GRID_SIZE[0]).reshape(-1, 1)
    for idx, ch in enumerate(text):
        phase = 2 * np.pi * ord(ch) / 256
        wave += np.exp(1j * (phase + 0.1 * idx))
    wave /= np.linalg.norm(wave)
    return wave


# --- Hilbert Space Extensions ---

def get_wavefunction():
    return psi_real + 1j * psi_imag

def normalize_wavefunction():
    global psi_real, psi_imag
    psi = get_wavefunction()
    norm = np.sqrt(np.sum(np.abs(psi)**2))
    if norm > 0:
        psi /= norm
        psi_real[:] = psi.real
        psi_imag[:] = psi.imag
    return norm

def inner_product(psi1_real, psi1_imag, psi2_real, psi2_imag):
    psi1 = psi1_real + 1j * psi1_imag
    psi2 = psi2_real + 1j * psi2_imag
    return np.vdot(psi1.flatten(), psi2.flatten())  # ⟨ψ1|ψ2⟩

def store_reference_state(name):
    psi = get_wavefunction()
    quantum_db[name] = {
        "real": psi.real.copy(),
        "imag": psi.imag.copy(),
        "timestamp": time.time()
    }
    print(f"💾 Stored quantum state as reference: '{name}'")

def compare_to_reference(name):
    if name not in quantum_db:
        print(f"❌ No reference named '{name}'")
        return
    ref = quantum_db[name]
    overlap = inner_product(psi_real, psi_imag, ref["real"], ref["imag"])
    magnitude = np.abs(overlap)
    print(f"🔎 Overlap with '{name}': |⟨ψ|ψ_ref⟩| = {magnitude:.4f}")
    return magnitude
def decode_wavefunction_to_text(n=None):
    return "<no prior encode>" if not last_encoded_text else last_encoded_text[:n] if n else last_encoded_text

# --- Animation ---
fig, ax = plt.subplots()
def animate(_):
    global tick
    evolve_wavefunction(DT)
    decay_field()
    step_agents()
    ax.clear()
    ax.imshow((psi_real ** 2 + psi_imag ** 2).T, cmap="plasma", origin="lower", alpha=0.7)
    ax.imshow(field.T, cmap="inferno", origin="lower", alpha=0.3)
    bx, by = np.where(barrier_mask)
    if bx.size:
        ax.scatter(bx, by, c="black", s=1)
    for a in agents:
        ax.plot(*a["pos"], "go", ms=3)
    tick += 1
    ax.set_title(f"Quantum OS Tick {tick}")

ani = animation.FuncAnimation(fig, animate, interval=50)

# --- Store & Show ---
def store_document(text, tag=""):
    doc_id = str(uuid.uuid4())[:8]
    ψ = encode_text_into_wavefunction(text)
    quantum_db[doc_id] = {
        "id": doc_id,
        "text": text,
        "tag": tag,
        "ts": time.time(),
        "wave": ψ
    }
    psi_real[:], psi_imag[:] = ψ.real, ψ.imag
    print(f"📦 Stored document {doc_id} with tag '{tag}'.")

def show_documents():
    if not quantum_db:
        print("<no documents>")
        return
    ψ = psi_real + 1j * psi_imag
    print(f"🌊 norm={np.linalg.norm(ψ):.4f}  max|ψ|={np.max(np.abs(ψ)):.3f}")
    for k, v in quantum_db.items():
        ts = time.strftime("%H:%M:%S", time.localtime(v["ts"]))
        preview = v["text"][:40].replace("\n", " ") + ("…" if len(v["text"]) > 40 else "")
        print(f"[{k}] ({ts}) Tag: '{v['tag']}' | Preview: {preview}")

# --- CLI ---
def cli_loop():
    print("🎧 Terminal ready. Type 'help' for commands.")
    while True:
        try:
            cmd = input("QOS > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting."); break
        if cmd == "help":
            print("""
setup single | setup slits | clear slits
emit single | emit slits
encode <text> | decode [n]
store <text> :: <tag>
show | exit
            """)
        elif cmd == "setup single":
            setup_single_slit(); print("🛠 Single slit setup.")
        elif cmd == "setup slits":
            setup_double_slit(); print("🛠 Double slit setup.")
        elif cmd == "clear slits":
            clear_barrier(); print("🧹 Barrier cleared.")
        elif cmd == "emit single":
            emit_single_slit_wave(); print("🎯 Single slit emission.")
            # Reposition agents in front of the single slit
            for agent in agents:
                agent['pos'] = np.array([GRID_SIZE[0] // 2 - 8 + np.random.uniform(-2, 2), GRID_SIZE[1] // 2 + np.random.uniform(-20, 20)], dtype=float)

        elif cmd == "emit slits":
            emit_two_slit_wave(); print("🎯 Double slit emission.")
            # Reposition agents at two positions aligned with the slits
            for i, agent in enumerate(agents):
                y_pos = GRID_SIZE[1] // 2 + (-10 if i % 2 == 0 else 10)
                agent['pos'] = np.array([GRID_SIZE[0] // 2 - 8 + np.random.uniform(-2, 2), y_pos + np.random.uniform(-15, 15)], dtype=float)

        elif cmd.startswith("encode "):
            text = cmd[7:]
            ψ = encode_text_into_wavefunction(text)
            psi_real[:] = ψ.real
            psi_imag[:] = ψ.imag
            print("🔐 Encoded text.")
        elif cmd.startswith("decode"):
            parts = cmd.split()
            length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            print("📜", decode_wavefunction_to_text(length))
        elif cmd.startswith("store "):
            body = cmd[6:]
            if "::" in body:
                text, tag = map(str.strip, body.split("::", 1))
            else:
                text, tag = body.strip(), ""
            store_document(text, tag)
        elif cmd == "show":
            show_documents()
        elif cmd == "exit":
            break
        else:
            print("❓ Unknown command. Type 'help'.")


# --- Quantum DB Functions ---
def store_document(text, tag):
    doc_id = str(uuid.uuid4())[:8]
    quantum_db[doc_id] = {"text": text, "tag": tag, "timestamp": time.time()}
    print(f"✅ Stored [{doc_id}]")

def show_documents():
    print("🧠 Quantum DB Contents:")
    for doc in quantum_db:
        preview = quantum_db[doc]["text"][:30].replace("\n", " ")
        print(f"  - [{doc}] Tag: '{quantum_db[doc]['tag']}' | Preview: '{preview}'")

def encode_text_into_wavefunction(text):
    # Naive encoding into real part of psi; can be upgraded with FFT or spatial phase modulation
    flat = np.frombuffer(text.encode(), dtype=np.uint8)
    flat = (flat - 127) / 255.0
    flat = np.pad(flat, (0, GRID_SIZE[0]*GRID_SIZE[1] - len(flat)), mode='constant')
    reshaped = flat[:GRID_SIZE[0]*GRID_SIZE[1]].reshape(GRID_SIZE)
    psi_real[:, :] = reshaped
    psi_imag[:, :] = 0
    return psi_real + 1j * psi_imag

def decode_wavefunction_to_text(length=None):
    flat = psi_real.flatten()
    byte_data = ((flat[:length or 1024] * 255) + 127).astype(np.uint8)
    try:
        return byte_data.tobytes().decode(errors='ignore')
    except:
        return "[Decode error]"
# --- Run Entry Point ---
def run():
    print("🚀 Launching Quantum OS…")
    plt.show(block=False)
    cli_loop()
    plt.close()

if __name__ == "__main__":
    run()


def emit_single_slit_wave():
    """
    Emits a Gaussian wave packet with a complex phase gradient to simulate a directional
    laser-like source aimed toward the double slit barrier. This is a more physically
    realistic representation of coherent photon or particle emission in experiments.
    """
    global psi_real, psi_imag

    x = np.linspace(0, GRID_SIZE[0], GRID_SIZE[0])
    y = np.linspace(0, GRID_SIZE[1], GRID_SIZE[1])
    X, Y = np.meshgrid(x, y)

    # --- Gaussian parameters ---
    center_x = GRID_SIZE[0] // 2 - 8  # Slightly behind the slit barrier
    center_y = GRID_SIZE[1] // 2      # Vertically centered
    sigma = 5.0                       # Width of the wave packet
    kx = 1.0                          # Forward momentum in x (controls phase gradient)
    ky = 0.0                          # No vertical initial motion

    # --- Gaussian amplitude ---
    amplitude = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))

    # --- Complex phase gradient (adds forward motion) ---
    phase = np.exp(1j * (kx * X + ky * Y))

    # --- Full wave packet ψ = A * exp(i(k·r)) ---
    psi = amplitude * phase

    # Separate into real and imaginary parts for simulation
    psi_real[:, :] = np.real(psi)
    psi_imag[:, :] = np.imag(psi)

def emit_general_wave():
    """
    Emits a simple, static real-valued Gaussian wave packet for general data simulations.
    This is used when quantum effects are recorded but no directional emission is needed.
    """
    global psi_real, psi_imag

    x = np.linspace(0, GRID_SIZE[0], GRID_SIZE[0])
    y = np.linspace(0, GRID_SIZE[1], GRID_SIZE[1])
    X, Y = np.meshgrid(x, y)

    # Centered wave
    center_x = GRID_SIZE[0] // 2
    center_y = GRID_SIZE[1] // 2
    sigma = 7.0

    psi_real[:, :] = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
    psi_imag[:, :] = 0

def cli_loop():
    print("🎧 Terminal ready. Type 'help' for commands.")
    while True:
        try:
            cmd = input("QOS > ").strip()
            if cmd == "exit":
                break
            elif cmd == "emit single":
                emit_single_slit_wave()
                print("🎯 Single slit emission.")
                for agent in agents:
                    agent['pos'] = np.array([
                        GRID_SIZE[0] // 2 - 4 + np.random.uniform(-1, 1),
                        GRID_SIZE[1] // 2 + np.random.uniform(-10, 10)
                    ], dtype=float)
            elif cmd == "emit general":
                emit_general_wave()
                print("📦 General wave emitted.")
                for agent in agents:
                    agent['pos'] = np.array([
                        GRID_SIZE[0] // 2 - 10 + np.random.uniform(-3, 3),
                        GRID_SIZE[1] // 2 + np.random.uniform(-30, 30)
                    ], dtype=float)
            
            elif cmd == "emit slits":
                emit_two_slit_wave()
                print("🎯 Double slit emission.")
                for i, agent in enumerate(agents):
                    # Move agents close to the barrier and align them with the slit centers
                    y_target = GRID_SIZE[1] // 2 - 12 if i % 2 == 0 else GRID_SIZE[1] // 2 + 12
                    agent['pos'] = np.array([
                        GRID_SIZE[0] // 2 - 4 + np.random.uniform(-1, 1),  # closer to the slits
                        y_target + np.random.uniform(-5, 5)
                    ], dtype=float)

            else:
                print("❓ Unknown command. Type 'help'.")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break