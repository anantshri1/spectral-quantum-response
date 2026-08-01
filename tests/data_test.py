import numpy as np
d = np.load("data/train.npz")
print("keys:", sorted(d.keys()))
# expect: u0_re_128, u0_im_128, uT_re_128, uT_im_128, V_128, (×256,512,1024),
#         p_x0, p_sigma, p_k0, p_coeffs
print("u0_re_128 shape:", d["u0_re_128"].shape)    # (2000, 128)
print("V_1024 shape:", d["V_1024"].shape)          # (2000, 1024)
print("p_coeffs shape:", d["p_coeffs"].shape)      # (2000, 64) complex

# spot-check the super-res invariant survived saving: reconstruct complex,
# confirm 128 == 1024[::8] on a random sample
i = 7
u0_128  = d["u0_re_128"][i]  + 1j*d["u0_im_128"][i]
u0_1024 = d["u0_re_1024"][i] + 1j*d["u0_im_1024"][i]
print("super-res invariant holds post-save:",
      np.allclose(u0_128, u0_1024[::8], atol=1e-10))   # should be True