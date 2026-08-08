# Derivative-Informed Operator Learning for Quantum Response Functions
A systematic study to test whether a Fourier Neural Operator (FNO) trained only on forward evolution `(ψ₀, V) → ψ_T` learns the correct response `∂ψ_T/∂V `as a byproduct.

---
## Background and Setup
The forward map only constrains the endpoint `ψ(T)`. The response is the propagator sandwiched around a perturbation and integrated over the whole trajectory:

```
δψ_T = -i ∫₀ᵀ U(T,t) · δV · U(t,0) · ψ₀ dt
```

Two potentials can produce nearly identical `ψ(T)` while having quite different response operators — the derivative sees the path, the endpoint does not. Therefore, assessing whether a model only sees the endpoints during training or also learns the correct path-sensitivity is a real empirical question.

### System setup

**1D free-rotor / quantum-pendulum problem**: a particle on a periodic ring (`circumference L=2π, ℏ=m=R=1`), Hamiltonian `H = -½∂²/∂φ² + V(φ)`, untwisted boundary conditions (`θ=0`, standard FFT applies directly — twisted BC would introduce Aharonov–Bohm/θ-vacuum physics we deliberately avoid). `ψ₀` is a random Gaussian wavepacket (center, width, integer carrier momentum `k₀∈[-4,4])`; V is a smooth Gaussian random field with a tunable spectral length-scale.

**Strang splitting**: K (kinetic) is diagonal in Fourier space, V (potential) is diagonal in real space, and `exp(-i(K+V)dt) ≠ exp(-iKdt)exp(-iVdt)` since `[K,V]≠0`. Symmetrizing as `exp(-iVdt/2)·exp(-iKdt)·exp(-iVdt/2)` cancels the leading commutator term, buying 2nd-order global accuracy for one extra FFT. Implemented as `jax.lax.scan` over timesteps rather than a Python for loop.

**Solver**: split-step Fourier (Strang splitting): `exp(-iVdt/2)·exp(-iKdt)·exp(-iVdt/2)`, implemented as `jax.lax.scan` over timesteps (not a Python loop — scan is what makes autodifferentiating through the ~10,000-step evolution tractable; a Python loop would make the response phase computationally impossible). Kinetic phase is a pointwise multiply in Fourier space (`k` from `fftfreq`, exact integers because `L=2π`); potential phase is a pointwise multiply in real space. Validated against four independent physics gates: exact plane-wave phase, norm conservation to float64 machine precision, analytic Gaussian-wavepacket spreading rate, and an autodiff-vs-finite-difference gradient check (clean U-shaped error valley bottoming at ~3×10⁻¹⁰, confirming `jax.jvp` through the scan is correct).

> Gate 1 — plane-wave eigenstate phase. With V=0, ψ=e^{ikφ} is an exact eigenstate of the kinetic operator, so one Strang step should multiply it by the analytically-known phase exp(-i·½k²·dt) with no shape change. Checked to atol=1e-5. This validates the FFT round-trip and the k-grid convention (jnp.fft.fftfreq(nx, d=1/nx), not arange — the top half of the array holds negative frequencies, e.g. bin 64 of a 128-point grid is k=-64 not k=+64, a silent-bug risk since the solver would run without error but with wrong kinetic energies).

> Gate 2 — norm conservation. Unitary evolution must conserve ∫|ψ|²dφ exactly. This tests the composed solver (all three Strang factors, many steps, real V) rather than individual factors. First run in float32 showed drift growing with nt at fixed total time (3.3e-5 → 3.4e-4 across nt=500→4000) — diagnosed via an nt-scaling probe as accumulated per-step FFT round-off, confirmed by re-running in float64 where drift collapsed to ~5e-14 and stayed flat regardless of nt. This result is what locked the precision split (solver/data-gen/response in float64, FNO training in float32) — decided from evidence, not assumed upfront, since unitary evolution has no dissipation to forgive float32 errors the way Burgers' viscosity did.

> Gate 3 — Gaussian wavepacket spreading. Tests genuine non-eigenstate dynamics (gate 1 only tested an eigenstate that doesn't change shape): a free packet's width should grow as σ(t)²=σ₀²+(t/2σ₀)². First attempt failed at t=0 (29% error before any evolution occurred), correctly diagnosed as a measurement bug rather than a solver bug, since nothing had run yet — traced to a √2 mismatch between the amplitude-width of the constructed ψ and the density-width (|ψ|²) the analytic formula assumes. A second attempt still failed at t=0 by the same √2 in the opposite direction, from over-correcting; resolved by collapsing to one single source of truth (pick amplitude-width a, feed the formula a/√2 and nothing else). Once fixed, t=0 agreed to ~1e-12 and dynamics tracked the analytic curve to ~1e-9, with error growing smoothly with t (expected accumulated round-off) and no sign of the ring's periodicity corrupting the result within the tested window.

> Gate 4 — gradient check (autodiff vs. finite differences). The gate that specifically de-risks Phase 4: since J_true will be computed by jax.jvp through the full scan, this validates that differentiation, not just evaluation, is correct. Central FD along a random direction δV, swept across ε∈[10⁻⁸,10⁻¹]. First single-point check (ε=1e-6) gave 1.9e-7 relative error — plausible but not sufficient evidence alone. A one-sided sweep (only small ε) showed monotonically rising error toward small ε, initially read as concerning, until recognized as only the round-off wall (∝1/ε, from amplifying fixed floating-point noise by dividing by shrinking ε) — the truncation wall (∝ε²) hadn't been sampled yet because this particular map is smooth enough that its valley floor sits at unusually large ε (~10⁻³) rather than the generic textbook estimate of ~10⁻⁵–10⁻⁶. Extending the sweep to large ε revealed the missing wall, giving a complete, textbook U-shaped valley bottoming at 2.8×10⁻¹⁰ with the correct power-law slope on each side (ε² falling, 1/ε rising) — the strongest form of confirmation, since a genuinely broken gradient would show a floor (FD converging to the true derivative while autodiff sits a fixed, non-vanishing distance away) rather than a valley reaching machine-precision-level agreement.

**Precision**: solver, data generation, and the "ground truth" response Jacobian all run in float64; FNO training runs in float32. This is a deliberate split, not an oversight — long unitary evolutions are precision-sensitive (no dissipation to forgive float32 round-off) and get differentiated through, while the FNO is a shallow, dissipation-tolerant learned function. Precision floor for `train.py` (f64-trained, canonical M=16 checkpoint) is forward rel-L2 = 0.0192, response error = 0.5787.

**FNO**: hand-implemented spectral convolution, `n_modes` as the swept architecture dial (2–16 in various experiments), complex `ψ` handled as two real channels at the model boundary (`w_real/w_imag` stored separately — complex-typed Adam second moments are ill-defined and destroy training).

**Response machinery** (`src/response.py`): `J_true = jacfwd` of `V ↦ ψ_T` (V held real, ψ₀ fixed — this differentiation is Wirtinger-clean because V is real, unlike differentiating w.r.t. complex ψ₀, which we deliberately avoid). `J_fno` is the identical machinery applied to the trained surrogate. Both are rotated from position space to mode space via FFT/IFFT to get `J(k,q) = ∂ψ̂_k/∂V̂_q`, interpreted as a physical mode-coupling operator (for a local potential, natively off-diagonal via `⟨k|V|q⟩ = V̂(k-q)`). `response_error = ‖J_fno - J_true‖_F / ‖J_true‖_F`.

---
## References
* **Derivative-informed neural operator acceleration of geometric MCMC for infinite-dimensional Bayesian inverse problems**, Lianghao Cao, Thomas O'Leary-Roseberry, Omar Ghattas. (2024). [arXiv: 2403.08220](https://arxiv.org/abs/2403.08220).
* **Derivative-Informed Operator Learning for Finance: On-the-Fly Greeks, Surfaces, Hedging, and Control**, Miquel Noguer I Alonso. (2026). [arXiv: 2606.05900](https://arxiv.org/abs/2606.05900).
* **Fourier Neural Operators for Learning Dynamics in Quantum Spin Systems**, Freya Shah, Taylor L. Patti, Julius Berner, Bahareh Tolooshams, Jean Kossaifi, Anima Anandkumar. (2024). [arXiv: 2409.03302](https://arxiv.org/abs/2409.03302).
* **The Propagator and the Path Integral**, Robert G. Littlejohn. (2021). [Notes](https://bohr.physics.berkeley.edu/classes/221/notes/pathint.pdf).
* **Introduction to Quantum Mechanics**, David Griffiths. (1995).
* **A particle on a ring or: how I learned to stop worrying and love θ-vacua**, Mohammad Aghaie, Ryosuke Sato. (2026). [arXiv: 2601.18248v1)](https://arxiv.org/abs/2601.18248v1).
* **Physical Chemistry for the Life Sciences**, Julio de Paula, Peter Atkins. (2006).
* **Time Evolution Split Operator Method**, Christina C. Lee. (2021). [Here](https://albi3ro.github.io/M4/Time-Evolution.html).
* **Exploring the Propagator of a Particle in a Box**, S. A. Fulling, K. S. Güntürk. [Here](https://people.tamu.edu/~fulling/box/box.pdf#:~:text=Here%20we%20take%20a%20close,wave%20function.).
* **Path Integral for a Particle in a Box**, A. Hershman. (1994). [Here](https://www.cds.caltech.edu/~marsden/wiki/uploads/projects/geomech/Hershman1994.pdf).
* **Path Integrals and Their Application to Dissipative Quantum Systems**, Gert-Ludwig Ingold. (2002). [arXiv: 0208026](https://arxiv.org/abs/quant-ph/0208026).
* **Spectral Neural Operators**, V. Fanaskov, I. Oseledets. (2022). [arXiv: 2205.10573](https://arxiv.org/abs/2205.10573).
* **Toward a Better Understanding of Fourier Neural Operators: Analysis and Improvement from a Spectral Perspective**, Shaoxiang Qin, Fuyuan Lyu, Wenhui Peng, Dingyang Geng, Ju Wang, Naiping Gao, Xue Liu, Liangzhu Leon Wang. (2024). [arXiv: 2404.07200](https://arxiv.org/abs/2404.07200v1).
* **Fourier Neural Operators Explained: A Practical Perspective**, Valentin Duruisseaux, Jean Kossaifi, Anima Anandkumar. (2024). [arXiv: 2512.01421](https://arxiv.org/abs/2512.01421).
* **Spectral Methods for Partial Differential Equations**, David Chopp. (2008). [Notes](https://people.esam.northwestern.edu/~chopp/course_notes/446-2.pdf).
* **LNO: Laplace Neural Operator for Solving Differential Equations**, Qianying Cao, Somdatta Goswami, George Em Karniadakis. (2023). [arXiv: 2303.10528](https://arxiv.org/abs/2303.10528).
* **A Library for Learning Neural Operators**, Jean Kossaifi, Nikola Kovachki, Zongyi Li, David Pitt, Miguel Liu-Schiaffini, Robert Joseph George, Boris Bonev, Kamyar Azizzadenesheli, Julius Berner, Valentin Duruisseaux, Anima Anandkumar. (2024). [arXiv: 2412.10354](https://arxiv.org/abs/2412.10354).
* **Neural Operator: Learning Maps Between Function Spaces**, Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. (2021). [arXiv: 2108.08481](https://arxiv.org/abs/2108.08481).
* **Redefining Neural Operators in d+1 Dimensions**, Haoze Song, Zhihao Li, Xiaobo Zhang, Zecheng Gan, Zhilu Lai, Wei Wang. (2025). [arXiv: 2505.11766v1](https://arxiv.org/abs/2505.11766v1).
* **Equinox: neural networks in JAX via callable PyTrees and filtered transformations**, Patrick Kidger, Cristian Garcia. (2021). [arXiv: 2111.00254](https://arxiv.org/abs/2111.00254).
* **Neural Operators: FNO and DeepONet**. [Watch here](https://www.youtube.com/watch?v=COEItKEZ-is).
* **Fourier Neural Operator for Parametric Partial Differential Equations**, Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. (2020). [arXiv: 2010.08895](https://arxiv.org/abs/2010.08895).
* **How a Fourier Neural Operator Learns to Solve PDEs — and Where It Falls Short**, Gurpreet Singh Hora, Prakhar Kapoor, Albert Matveev. (2026). [Here](https://www.physicsx.ai/newsroom/how-a-fourier-neural-operator-learns-to-solve-pdes----and-where-it-falls-short).



