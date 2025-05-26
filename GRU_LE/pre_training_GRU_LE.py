# ================================================================
#  pre_training_GRU_LE.py  –  exact SMNIST-GRU pipeline (Vogt+24)
# ================================================================
"""
Train a 1-layer GRU on *row-wise* Sequential-MNIST (SMNIST) and compute
its full Lyapunov spectrum, faithfully matching the protocol in
Vogt et al., “Lyapunov-Guided Representation …” (arXiv:2204.04876).

# find the critical gain (watch the print-out for sign change)
python pre_training_GRU_LE.py --calibrate --device cuda

# pick the zero-crossing (say g* ≈ 1.25) and fix it below
python pre_training_GRU_LE.py --gain 1.27 --device cuda
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse, math, os, random, pathlib, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
import torch, torch.nn as nn
from math import sqrt, log


# Data utilities
def get_loaders(batch=128, root="./torch_datasets"):
    tfm = transforms.ToTensor()
    root = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    ds_te = MNIST(root, train=False, download=True, transform=tfm)
    n_tr  = int(0.8 * len(ds_tr))               # 80 % train / 20 % val
    ds_tr, ds_va = random_split(ds_tr, [n_tr, len(ds_tr) - n_tr])
    def mk(ds, shuf): return DataLoader(ds, batch, shuffle=shuf, drop_last=True)
    return mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)


# Model
class GRUSMNIST(nn.Module):
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.gru    = nn.GRU(28, hidden, batch_first=True, bias=False)   # 28 × 28 → 28-step row stream
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden, 10, bias=False)

    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, 28)             # row-wise unfold
        h0  = torch.zeros(1, B, self.hidden, device=x.device, dtype=x.dtype)
        y, _ = self.gru(seq, h0)
        y    = self.drop(y)                 # dropout matches repo
        return self.fc(y[:, -1])


# Initialization
def critical_gru_init(model: nn.Module,
                      g: float = 1.0,
                      scheme: str = "orthogonal"):
    """
    Random-matrix–consistent init for a 1-layer GRU.
    Args
    ----
    g      : gain controlling spectral radius (≈1 for edge-of-chaos)
    scheme : 'orthogonal' | 'gaussian'
    """
    H = model.hidden
    gain = g / sqrt(H)

    # -- weight_hh -------------------------------------------------
    if scheme == "orthogonal":
        with torch.no_grad():                                # NEW
            nn.init.orthogonal_(model.gru.weight_hh_l0)
            model.gru.weight_hh_l0.mul_(g)                   # safe now
    else:
        nn.init.normal_(model.gru.weight_hh_l0, 0.0, gain)

    # -- weight_ih -------------------------------------------------
    nn.init.normal_(model.gru.weight_ih_l0, 0.0, gain)

    # -- gate biases ----------------------------------------------
    if hasattr(model.gru, "bias_hh_l0"):
        with torch.no_grad():
            model.gru.bias_hh_l0.zero_()
            model.gru.bias_ih_l0.zero_()
            model.gru.bias_hh_l0[H:2*H] = -0.1     # small neg. update-gate bias

    return g



# Training loop
def run_epoch(net, loader, crit, opt, train, device, desc):
    net.train(train)
    tot_loss = 0.0; correct = 0; seen = 0
    for X, y in tqdm(loader, desc=desc, leave=False):
        X, y = X.to(device), y.to(device)
        if train: opt.zero_grad()
        logit = net(X)
        loss  = crit(logit, y)
        if train:
            loss.backward(); clip_grad_norm_(net.parameters(), 5.0); opt.step()
        tot_loss += loss.item() * X.size(0)
        correct  += (logit.argmax(1) == y).sum().item()
        seen     += X.size(0)
    return tot_loss / seen, correct / seen


#  Jacobian via autograd.functional.jacobian  (no batch dimension)
def gru_jacobian_autograd(cell: nn.GRUCell,
                          x_t: torch.Tensor,      # shape (28,)
                          h_prev: torch.Tensor):  # shape (H,)
    """
    Compute J = ∂h_t / ∂h_{t-1} using PyTorch autograd.
    Returns a (H,H) tensor on the same device/dtype as h_prev.
    """
    # Ensure h_prev participates in the graph
    h_prev = h_prev.detach().requires_grad_(True)

    # Closure: only h is treated as input — x_t is constant
    def _func(h):
        return cell(x_t, h)

    # jacobian returns shape (H, H)
    J = jacobian(_func, h_prev, create_graph=False, strict=True)  # reverse-mode
    return J.detach()


#  Lyapunov spectrum using autograd Jacobian
def lyap_spectrum(model,
                  seq: torch.Tensor,           # (T, 28)
                  *, warm: int = 500, progress=True) -> torch.Tensor:
    """
    Compute the full Lyapunov spectrum of a trained 1-layer GRU on an
    input sequence seq.  Jacobians are obtained with autograd.
    Returns tensor (H,) on CPU.
    """
    H   = model.hidden
    dev = seq.device
    dty = seq.dtype

    # Build a matching GRUCell and copy weights
    cell = nn.GRUCell(28, H, bias=False, device=dev, dtype=dty)
    cell.load_state_dict({
        'weight_ih': model.gru.weight_ih_l0,
        'weight_hh': model.gru.weight_hh_l0
    },strict=False)

    # Containers
    h   = torch.zeros(H, device=dev, dtype=dty)
    Q   = torch.eye(H, device=dev, dtype=dty)
    le_sum = torch.zeros(H, device=dev, dtype=dty)
    steps  = 0

    # Warm-up phase (no LE accumulation)
    for t in range(warm):
        h = cell(seq[t], h)

    # Main QR loop
    iterator = range(warm, seq.size(0))
    if progress:
        iterator = tqdm(iterator, desc="Lyap-QR", leave=False)
    for t in iterator:
        J = gru_jacobian_autograd(cell, seq[t], h)      # (H,H)
        Q, R = torch.linalg.qr(J @ Q)                   # reduced QR
        #le_sum += torch.log(torch.abs(torch.diagonal(R)))
        eps = 1e-12
        le_sum += torch.log(torch.clamp(torch.abs(torch.diagonal(R)), min=eps))
        h = cell(seq[t], h)
        steps += 1

    return (le_sum / steps).cpu()                       # (H,)


def make_le_driver(batch=15, seq_len=100, device='cpu', dtype=torch.float64):
    """
    Return a tensor (batch, T, 28) of i.i.d. U(0,1) noise for LE calculation.
    Matches the ‘one-hot / random’ driver used in Vogt et al. (2024).
    """
    return torch.rand(batch, seq_len, 28, device=device, dtype=dtype)

# --------------------------------------------------------------
# Fast single-column QR to estimate only λ_max (≪ full spectrum)
# --------------------------------------------------------------
def lambda_max(net: nn.Module,
               warm: int = 500, T: int = 100,
               device: str = "cuda") -> float:
    torch.set_default_dtype(torch.float64)
    net = net.double().to(device)

    driver = make_le_driver(batch=1, seq_len=warm + T,
                            device=device, dtype=torch.float64)[0]

    H = net.hidden
    cell = nn.GRUCell(28, H, bias=False, device=device, dtype=torch.float64)
    cell.load_state_dict({"weight_ih": net.gru.weight_ih_l0,
                          "weight_hh": net.gru.weight_hh_l0}, strict=False)

    h = torch.zeros(H, dtype=torch.float64, device=device)
    q = torch.randn(H, 1, dtype=torch.float64, device=device)
    q /= q.norm()

    log_r_sum = 0.0
    for t in range(warm + T):
        h = cell(driver[t], h)
        if t < warm:
            continue
        J = gru_jacobian_autograd(cell, driver[t], h)
        v = J @ q
        r = v.norm()
        q = v / (r + 1e-12)
        log_r_sum += torch.log(r + 1e-12)

    return (log_r_sum / T).item()

# --------------------------------------------------------------
# Calibrate g  →  λ_max curve
# --------------------------------------------------------------
def calibrate(hidden, g_grid, scheme, device, n_drivers=5, driver_T=200):
    λs = []
    for g in g_grid:
        vals = []
        for _ in range(n_drivers):             # average → smoother curve
            net = GRUSMNIST(hidden).to(device)
            critical_gru_init(net, g=g, scheme=scheme)
            vals.append(lambda_max(net, T=driver_T, device=device))
        λ = np.mean(vals);  λs.append(λ)
        print(f"gain {g:4.2f}  →  λ_max = {λ:+7.4f}")

    # plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,3))
    plt.plot(g_grid, λs, marker="o")
    plt.axhline(0, color="k", ls="--", lw=.8)
    plt.xlabel("gain  g")
    plt.ylabel(r"$\lambda_{\max}$")
    plt.title("Critical-gain calibration")
    plt.tight_layout()
    plt.savefig("lambda_max_vs_gain.png", dpi=200)
    print("saved  lambda_max_vs_gain.png")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch',  type=int, default=128)
    ap.add_argument('--lr',     type=float, default=1e-2)
    ap.add_argument('--trials', type=int, default=40,
                    help='number of independent runs to average')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--calibrate', action='store_true',
                    help='run g→λ_max sweep and exit')
    ap.add_argument('--gain', type=float, default=1.05,
                help='initialisation gain g (ignored when --calibrate)')
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.calibrate:                    # <-- early-exit branch
        calibrate(args.hidden,
                  g_grid=np.arange(0.5, 9, 0.5),
                  scheme="gaussian",
                  device=device,
                  n_drivers=30,                         # new arg
                  driver_T=400)                        # longer window
        return

    _, _, teL = get_loaders(args.batch)

    all_LE = []        # will collect one (H,) array per trial
    all_g  = []        # keep the p that was drawn each run

    for run in range(1, args.trials + 1):
        # build & init model
        net = GRUSMNIST(args.hidden).to(device)
        g = critical_gru_init(net, g=4.0, scheme="gaussian")
        all_g.append(g)

        print(f"\n=== Trial {run}/{args.trials}  (g = {g:.3f}) ===")

        # (after training) Lyapunov-Exponent driver – 15 random sequences
        torch.set_default_dtype(torch.float64)  # double precision like the paper
        net = net.double()

        WARM = 500              # Warmup
        SEQ  = 100              # length of the window you’ll average over
        driver = make_le_driver(batch=15, seq_len=WARM + SEQ,
                                device=device, dtype=torch.float64)

        LE_batch = []
        for seq in driver:                      # iterate over the 15 sequences
            LE = lyap_spectrum(net, seq, warm=WARM, progress=True)  # ← warm-up!
            LE_batch.append(LE.cpu().numpy())

        LE = np.mean(LE_batch, axis=0)          # mean over the 15 spectra
        # ------------------------------------------------------------------


        all_LE.append(LE)                           # good run → keep
        np.save(f"lyap_spectrum_T{run}.npy", LE)
        print(f"  λ₁ = {LE[0]:+.6f}   λ_H = {LE[-1]:+.6f}")


    # average over trials
    #mean_LE = np.mean(all_LE, axis=0)
    if len(all_LE) == 0:
        raise RuntimeError("Every trial was discarded – no spectrum to average.")
    mean_LE = np.mean(all_LE, axis=0)

    np.save("lyap_spectrum_mean.npy", mean_LE)

    print("\n===  A v e r a g e  over 100 trials  ===")
    print(f"λ₁̄ = {mean_LE[0]:+.6f}   λ̄_H = {mean_LE[-1]:+.6f}")
    print(f"g values drawn: {', '.join(f'{x:.3f}' for x in all_g)}")

    # plot mean spectrum
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.5,3.2))
    plt.plot(range(1, len(mean_LE)+1), mean_LE,
             marker='o', markersize=3, lw=1.5, label='mean of 200 trials')
    plt.axhline(0, color='black', lw=.8, ls='--')
    plt.xlabel('Exponent index  $i$')
    plt.ylabel(r'$\bar{\lambda}_i$')
    plt.title('Mean Lyapunov spectrum (200 trials)')
    plt.tight_layout()
    plt.savefig('lyap_spectrum_mean.png', dpi=300)
    print("Saved  lyap_spectrum_mean.npy  and  lyap_spectrum_mean.png")

if __name__ == '__main__':
    main()