from dataclasses import dataclass
from math import pi

@dataclass
class Config:
    # --- Domain (LOCKED) ---
    L: float = 2 * pi      # periodic box. L=2π makes wavenumbers integers: k = n.
    nx: int = 128          # grid points. Full J is 128×128 — trivially materializable.
    nx_mid1: int = 256
    nx_mid2: int = 512
    nx_super: int = 1024
    nt: int = 1000         # solver timesteps. Strang is 2nd order; scale nt with nx.
    nt_super: int = 10000 
    t_end: float = 0.5     # long enough for off-diagonal coupling,
                           # short enough that forward stays learnable + phase error small.

    # --- ψ0 family: Gaussian wavepackets ---
    psi0_x0_lo: float = 0     
    psi0_x0_hi: float = 2*pi
    psi0_sigma_lo: float = 0.3  # small vs L so the packet ≈0 at boundaries
    psi0_sigma_hi: float = 0.6
    psi0_k0_lo: int = -4        # carrier momentum; integers keep it single-valued on the box
    psi0_k0_hi: int = 4         # band-limited => forward is learnable

    # --- V family: smooth GRF (this is the complexity/OOD dial) ---
    v_length_scale: float = 0.2   # smaller = more high-k structure = harder response
    v_amplitude: float = 1.0      # keep V_max * dt << 1 for Strang phase accuracy
    v_n_coeffs: int = 64

    # --- Data ---
    n_train_max: int = 2000
    n_test: int = 200
    n_probe: int = 300            # Phase 3 forward-learnability probe (cheap, pre-grid)

    # --- FNO architecture ---
    in_channels: int = 4     # Re ψ0, Im ψ0, V, grid   (LOCKED by the problem)
    out_channels: int = 2    # Re ψ_T, Im ψ_T          (LOCKED)
    n_modes: int = 16        # truncation — THE dial the whole project studies
    n_channels: int = 32
    n_fno_blocks: int = 4

    # --- Training ---
    n_epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 3e-4   # matches the corrected Burgers value
    lr_decay_rate: float = 0.5
    lr_decay_every: int = 100
    seed: int = 0

    # --- Experiment ---
    data_dir: str = "data"
    experiment_name: str = "spectral-quantum-response"