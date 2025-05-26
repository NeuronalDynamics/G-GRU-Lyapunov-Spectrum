import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV that your sweep saved (make sure the file is in the working directory)
df = pd.read_csv("./phase_diagram.csv")

# --- 1.2  λ₁ mean ± std -------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 3))
for width, g in df.groupby("width"):
    stats = g.groupby("p").lambda_max.agg(["mean", "std"]).reset_index()
    ax1.errorbar(stats["p"], stats["mean"], yerr=stats["std"],
                 fmt='o-', label=f"H={width}")

ax1.axhline(0, lw=.8, ls='--')
#ax1.set_xscale("log")
ax1.set_xlabel("initial scale p")
ax1.set_ylabel(r"lambda_max  (± std)")
ax1.set_title("Mean ± std of largest Lyapunov exponent")
ax1.legend()
plt.tight_layout()
plt.show()

# --- 1.3  phase–probability curve --------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 3))
crit_points = {}
for width, g in df.groupby("width"):
    prob = g.groupby("p").apply(lambda x: (x.lambda_max > 0).mean()).reset_index(name="prob")
    ax2.plot(prob["p"], prob["prob"], marker='o', label=f"H={width}")
    
    # interpolate p* where prob ≈ 0.5
    p_star = np.interp(0.5, prob["prob"], prob["p"])
    crit_points[width] = p_star

#ax2.set_xscale("log")
ax2.set_xlabel("initial scale $p$")
ax2.set_ylabel("Pr[lambda_max > 0]")
ax2.set_title("Probability of chaos vs. scale\n(interpolated p* where Pr=0.5)")
ax2.axhline(0.5, lw=.8, ls='--')
ax2.legend()
plt.tight_layout()
plt.show()

# Print the interpolated critical p*
for H, p_star in crit_points.items():
    print(f"Estimated p* for H={H}: {p_star:.3f}")
