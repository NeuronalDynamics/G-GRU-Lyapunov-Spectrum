import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 0.  Load sweep results
# --------------------------------------------------------------
df = pd.read_csv("./phase_diagram.csv")

# --------------------------------------------------------------
# A.  λ₁ mean ± std            (unchanged)
# --------------------------------------------------------------
summary = (
    df.groupby(["width", "p"]).lambda_max.agg(["mean", "std"]).reset_index()
)

print("\n=====  λ₁ statistics  (mean ± std)  =====")
print(summary.to_string(
        index=False,
        formatters={
            "width": "{:>3d}".format,
            "p":     "{:.3f}".format,
            "mean":  "{:+.5f}".format,
            "std":   "{:.5f}".format,
        })
)

# --------------------------------------------------------------
# B.  Plot mean ± std          (unchanged)
# --------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 3))
for width, g in summary.groupby("width"):
    ax1.errorbar(g["p"], g["mean"], yerr=g["std"],
                 fmt='o-', label=f"H={width}")
ax1.axhline(0, lw=.8, ls='--')
ax1.set_xscale("log")
ax1.set_xlabel("initial scale $p$")
ax1.set_ylabel(r"$\lambda_{\max}$  (mean ± std)")
ax1.set_title("Largest Lyapunov exponent")
ax1.legend(title="hidden size")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# C.  Probability of chaos  — NO MORE WARNING
#     * avoid apply() entirely *
# --------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 3))
crit_points = {}

for width, g in df.groupby("width"):
    # Boolean mask: λ₁ > 0
    chaos = g.assign(is_chaotic = g["lambda_max"] > 0)

    # mean of the Boolean gives the probability
    prob = (
        chaos.groupby("p")["is_chaotic"]
             .mean()                 # ← aggregation, no apply()  ✔
             .reset_index(name="prob")
    )

    ax2.plot(prob["p"], prob["prob"], marker='o', label=f"H={width}")

    # interpolate p* where prob ≈ 0.5
    p_star = np.interp(0.5, prob["prob"], prob["p"])
    crit_points[width] = p_star

ax2.set_xscale("log")
ax2.set_xlabel("initial scale $p$")
ax2.set_ylabel(r"$\Pr[\lambda_{\max}>0]$")
ax2.set_title("Probability of chaos vs. scale")
ax2.axhline(0.5, lw=.8, ls='--')
ax2.legend(title="hidden size")
plt.tight_layout()
plt.show()

print("\n=====  interpolated p* (P≈0.5)  =====")
for H, p_star in crit_points.items():
    print(f"H={H:>3d}:  p* ≈ {p_star:.4f}")
