# ================================================================
#  pre_training_G-GRU_LE.py
# ================================================================
"""
Train a 1-layer G-GRU on *row-wise* Sequential-MNIST (SMNIST) and compute
its full Lyapunov spectrum

python pre_training_G-GRU_LE.py --model ggru --device cuda
python pre_training_G-GRU_LE.py --model gru  --calibrate --device cuda

# find the critical gain (watch the print-out for sign change)
python pre_training_G-GRU_LE.py --model ggru --k 4 --calibrate --device cuda

# pick the zero-crossing (say g* ≈ 1.25) and fix it below
python pre_training_G-GRU_LE.py --model ggru --k 4 --gain 8.0 --trials 30 --device cuda
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
import copy


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

# ================================================================
#  ⇒⇒ NEW: helpers for block-constant permutation-equivariant GRU
# ================================================================
def _expand_block(a_diag, a_off, k):
    """
    Build a (k,k) block-constant matrix with diag=a_diag and off-diag=a_off.
    Returns tensor of shape (k, k).
    """
    dev, dtype = a_diag.device, a_diag.dtype
    mat = a_off.expand(k, k).clone()
    idx = torch.arange(k, device=dev)
    mat[idx, idx] = a_diag
    return mat


class GGRUCell(nn.Module):
    """
    Gordon-Lopez-Paz-Baroni permutation-equivariant GRU cell (ICLR 2020).
    Hidden units are split into `k` orbits of size H//k.
    Each gate weight is W = A ⊗ I + B ⊗ (1_k 1_kᵀ − I)  (block-constant).
    """
    def __init__(self, input_size, hidden_size, k, bias=False):
        super().__init__()
        assert hidden_size % k == 0, "hidden_size must be divisible by k"
        self.k, self.h = k, hidden_size // k
        self.input_size = input_size
        self.hidden_size = hidden_size
        # base blocks for each gate: reset r, update z, candidate n
        self.A_hh = nn.Parameter(torch.Tensor(3, self.h, self.h))
        self.B_hh = nn.Parameter(torch.Tensor(3, self.h, self.h))
        self.W_ih  = nn.Parameter(torch.Tensor(3*hidden_size, input_size))
        if bias:
            self.bias_hh = nn.Parameter(torch.zeros(3*hidden_size))
            self.bias_ih = nn.Parameter(torch.zeros(3*hidden_size))
        else:
            self.register_parameter("bias_hh", None)
            self.register_parameter("bias_ih", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialise later via critical_ggru_init
        nn.init.zeros_(self.A_hh)
        nn.init.zeros_(self.B_hh)
        nn.init.zeros_(self.W_ih)

    # ───────────────────────────────────────────────────────────────
    def _weight_hh_full(self):
        """
        Return a (3H, H) tensor with
            W = A ⊗ I_k  +  B ⊗ (1_k 1_kᵀ − I_k)
        for each gate (reset r, update z, candidate n).
        """
        k, h = self.k, self.h
        dev, dty = self.A_hh.device, self.A_hh.dtype

        eye_k   = torch.eye(k,  device=dev, dtype=dty)
        ones_k  = torch.ones(k, device=dev, dtype=dty)
        off_mat = ones_k[:, None] @ ones_k[None, :] - eye_k        # Jₖ − Iₖ

        blocks = []
        for g in range(3):                                         # r, z, n
            A, B = self.A_hh[g], self.B_hh[g]                      # (h,h)
            Wg = torch.kron(eye_k, A) + torch.kron(off_mat, B)     # (H,H)
            blocks.append(Wg)
        return torch.cat(blocks, dim=0)                            # (3H, H)
    # ───────────────────────────────────────────────────────────────

    def forward(self, x_t, h_prev):
        # ------------------------------------------------------------
        # NEW: allow either (H_in) or (B, H_in) inputs   ← ★★★★★
        squeeze = False
        if x_t.dim() == 1:          # driver in λ_max / lyap_spectrum
            x_t = x_t.unsqueeze(0)  # (1, input_size)
            h_prev = h_prev.unsqueeze(0)
            squeeze = True
        # ------------------------------------------------------------

        W_hh   = self._weight_hh_full()            # (3H, H)
        hh_lin = torch.matmul(h_prev, W_hh.T)      # (B, 3H)
        ih_lin = torch.matmul(x_t, self.W_ih.T)    # (B, 3H)

        if self.bias_hh is not None:
            hh_lin = hh_lin + self.bias_hh
            ih_lin = ih_lin + self.bias_ih

        r, z, n = torch.split(hh_lin + ih_lin, self.hidden_size, dim=-1)
        r = torch.sigmoid(r);  z = torch.sigmoid(z)
        n = torch.tanh(r * n)
        h = (1 - z) * n + z * h_prev

        return h.squeeze(0) if squeeze else h      # keep old API


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

# ================================================================
#  ⇒⇒ NEW: permutation-equivariant SMNIST model (G-GRU)
# ================================================================
class GGRUSMNIST(nn.Module):
    def __init__(self, hidden=64, k=4, dropout=0.1):
        super().__init__()
        self.hidden, self.k = hidden, k
        self.cell = GGRUCell(28, hidden, k, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 10, bias=False)

    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, 28)
        h   = torch.zeros(B, self.hidden, device=x.device, dtype=x.dtype)
        for t in range(28):
            h = self.cell(seq[:, t], h)
        h = self.drop(h)
        return self.fc(h)


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


def critical_ggru_init(model: GGRUSMNIST, g_star, k, scheme="orthogonal"):
    """
    Initialise base blocks so *effective* σ matches vanilla GRU.
    Because var(W_hh|_{gate}) = (A² + (k-1)B²)/k, choosing A = B
    makes that variance equal to B².  Since σ = g_star / √H already
    contains the 1/√H factor, **do not multiply by √k any more.**
    """
    H = model.hidden; h = H // k
    #sigma = g_star * math.sqrt(k) / math.sqrt(H)
    sigma = g_star / sqrt(H)
    with torch.no_grad():
        if scheme == "orthogonal":
            nn.init.orthogonal_(model.cell.A_hh); model.cell.A_hh.mul_(sigma)
            #nn.init.orthogonal_(model.cell.B_hh); model.cell.B_hh.mul_(sigma)
            model.cell.B_hh.copy_(model.cell.A_hh); model.cell.B_hh.mul_(sigma)
        else:
            nn.init.normal_(model.cell.A_hh, 0.0, sigma)
            #nn.init.normal_(model.cell.B_hh, 0.0, sigma)
            model.cell.B_hh.copy_(model.cell.A_hh)        # A = B  ❰★❱
        nn.init.normal_(model.cell.W_ih, 0.0, sigma)
    return g_star


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


# Generic Jacobian  (works for GRUCell *or* GGRUCell)--
def rnn_jacobian_autograd(cell: nn.Module,
                          x_t: torch.Tensor,      # (input_dim,)
                          h_prev: torch.Tensor):  # (H,)
    """
    Returns J = ∂h_t / ∂h_{t-1} as an (H,H) tensor on the same device/dtype.
    Works for any cell whose forward signature is   cell(x_t, h_prev).
    """
    h_prev = h_prev.detach().requires_grad_(True)

    def _func(h):
        return cell(x_t, h)

    J = jacobian(_func, h_prev, create_graph=False, strict=True)
    return J.detach()


#  Lyapunov spectrum using autograd Jacobian
def lyap_spectrum(model, seq, *, warm=500, progress=True):
    H, dev, dty = model.hidden, seq.device, seq.dtype

    # ----------------------------------------------------------
    # NEW: cell clone for GRU or G-GRU
    # ----------------------------------------------------------
    if hasattr(model, "gru"):
        cell = nn.GRUCell(28, H, bias=False, device=dev, dtype=dty)
        cell.load_state_dict({"weight_ih": model.gru.weight_ih_l0,
                              "weight_hh": model.gru.weight_hh_l0},
                             strict=False)
    else:
        cell = copy.deepcopy(model.cell).to(dev).to(dty)
    # ----------------------------------------------------------

    h   = torch.zeros(H, device=dev, dtype=dty)
    Q   = torch.eye(H, device=dev, dtype=dty)
    le_sum = torch.zeros(H, device=dev, dtype=dty)
    steps  = 0

    for t in range(warm):         # warm-up
        h = cell(seq[t], h)

    iterator = range(warm, seq.size(0))
    if progress:
        iterator = tqdm(iterator, desc="Lyap-QR", leave=False)
    for t in iterator:
        # standard QR step
        J = rnn_jacobian_autograd(cell, seq[t], h)
        Q, R = torch.linalg.qr(J @ Q)
        # periodic re-orthonormalisation (Benettin refresh)
        if steps % 50 == 0:                 # every 50 steps is plenty
            Q, _ = torch.linalg.qr(Q)       # keep Q well-conditioned
            Q, R = torch.linalg.qr(J @ Q)   # recompute current R
        # accumulate logarithms
        le_sum += torch.log(torch.clamp(torch.abs(torch.diagonal(R)), 1e-12))
        h = cell(seq[t], h)
        steps += 1

    return (le_sum / steps).cpu()



def make_le_driver(batch=15, seq_len=100, device='cpu', dtype=torch.float64):
    """
    Return a tensor (batch, T, 28) of i.i.d. U(0,1) noise for LE calculation.
    Matches the ‘one-hot / random’ driver used in Vogt et al. (2024).
    """
    return torch.rand(batch, seq_len, 28, device=device, dtype=dtype)


# ================================================================
#  ⇒⇒  PATCHED equivariance test  (place after model definitions)
# ================================================================
def check_equivariance(model: GGRUSMNIST, *, tol: float = 1e-6):
    """
    Permute k hidden orbits with P⊗I_h and verify f(x) = f^P(x).
    """
    # 1️⃣ deterministic, DOUBLE-precision clones  <<<<<<<<  NEW
    model = copy.deepcopy(model).eval().double()        # **NEW**
    k, h   = model.k, model.hidden // model.k
    dev    = next(model.parameters()).device
    I_h    = torch.eye(h, dtype=torch.float64, device=dev)  # **NEW**

    P_k    = torch.eye(k, dtype=torch.float64, device=dev)[torch.randperm(k, device=dev)]
    P_big  = torch.kron(P_k, I_h)                         # (H,H)
    P_full = torch.block_diag(P_big, P_big, P_big)        # (3H,3H)

    torch.manual_seed(0)
    x = torch.randn(2, 1, 28, 28, dtype=torch.float64, device=dev)  # **NEW**
    y_ref = model(x)

    mp = copy.deepcopy(model).eval()
    mp.fc.weight.data = mp.fc.weight @ P_big.T
    mp.cell.W_ih.data = P_full @ mp.cell.W_ih
    torch.manual_seed(0)
    y_perm = mp(x)

    assert torch.allclose(y_ref, y_perm, atol=tol), "equivariance broken!"





# --------------------------------------------------------------
# Fast single-column QR – now model-agnostic
# --------------------------------------------------------------
def lambda_max(net: nn.Module,
               warm: int = 500, T: int = 100,
               device: str = "cuda") -> float:
    torch.set_default_dtype(torch.float64)
    net = net.double().to(device)

    drivers = make_le_driver(batch=5, seq_len=warm + T,
                            device=device, dtype=torch.float64)

    H = net.hidden
    # ------------------------------------------------------------------
    # NEW: choose the correct cell implementation
    # ------------------------------------------------------------------
    if hasattr(net, "gru"):                     # vanilla GRU model
        cell = nn.GRUCell(28, H, bias=False,
                          device=device, dtype=torch.float64)
        cell.load_state_dict({"weight_ih": net.gru.weight_ih_l0,
                              "weight_hh": net.gru.weight_hh_l0},
                             strict=False)
    else:                                       # permutation-equivariant model
        cell = copy.deepcopy(net.cell).to(device).double()  # freeze clone
    # ------------------------------------------------------------------

    h = torch.zeros(H, dtype=torch.float64, device=device)
    q = torch.randn(H, 1, dtype=torch.float64, device=device)
    q /= q.norm()

    log_r_sum = 0.0
    for drv in drivers:              # loop over each random sequence
        for t in range(warm + T):
            h = cell(drv[t], h)
            if t < warm:
                continue
            J = rnn_jacobian_autograd(cell, drv[t], h)
            v = J @ q
            r = v.norm()
            q = v / (r + 1e-12)
            log_r_sum += torch.log(r + 1e-12)

    return (log_r_sum / (T * len(drivers))).item()

def calibrate_full(net, g0, eps=2e-3, k=4, max_iter=8):
    g = g0
    for _ in range(max_iter):
        lam = lambda_max(net, T=600)
        if abs(lam) < eps: break
        # secant update: shift by lam / |∂λ/∂g|  (≈ 0.25 σ for GRU)
        g -= 0.8 * lam * sqrt(k)
        net = GGRUSMNIST(net.hidden, k=k).to(net.fc.weight.device)
        critical_ggru_init(net, g_star=g, k=k, scheme='gaussian')
    return g

# --------------------------------------------------------------
# Calibrate g  →  λ_max curve
# --------------------------------------------------------------
def calibrate(hidden, g_grid, scheme, device,
              model_type="gru", k=4,
              n_drivers=5, driver_T=200):
    λs = []
    for g in g_grid:
        vals = []
        for _ in range(n_drivers):
            # -----------------------------------------
            # NEW: pick model + init according to flag
            # -----------------------------------------
            if model_type == "gru":
                net = GRUSMNIST(hidden).to(device)
                critical_gru_init(net, g=g, scheme=scheme)
            else:                                  # G-GRU
                net = GGRUSMNIST(hidden, k=k).to(device)
                critical_ggru_init(net, g_star=g, k=k, scheme=scheme)
            # (optional) quick equivariance check once
            if model_type == 'ggru' and _ == 0:
                check_equivariance(net)
                g = calibrate_full(net, g, k=k)   # secant refinement
                # rebuild net with refined gain
                net = GGRUSMNIST(hidden, k=k).to(device)
                critical_ggru_init(net, g_star=g, k=k, scheme=scheme)
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
    ap.add_argument('--gain', type=float, default=3.22,
                help='initialisation gain g (ignored when --calibrate)') #4.0 for GRU #3.22 for GGRU k=4 #2.06 for GGRU k=8
    ap.add_argument('--model', choices=['gru', 'ggru'], default='gru')
    ap.add_argument('--k', type=int, default=4,
                    help='number of permutation orbits (for ggru)') #4 8

    args = ap.parse_args()

    device = torch.device(args.device)

    if args.calibrate:
        if args.model == 'ggru':
            #g_grid = np.arange(2.00, 2.3, 0.05)
            g_grid = np.arange(3.20, 3.30, 0.01)
        else:
            g_grid = np.arange(3.9, 4.1, 0.01)

        print(g_grid)
        calibrate(hidden=args.hidden,
                  g_grid=g_grid,
                  scheme="gaussian",
                  device=device,
                  model_type=args.model,
                  k=args.k,
                  n_drivers=3,
                  driver_T = 1500 if args.k >= 8 else 800)
        return

    _, _, teL = get_loaders(args.batch)

    all_LE = []        # will collect one (H,) array per trial
    all_g  = []        # keep the p that was drawn each run

    for run in range(1, args.trials + 1):
        # build & init model
        #net = GRUSMNIST(args.hidden).to(device)
        #g = critical_gru_init(net, g=4.0, scheme="gaussian")
        if args.model == 'gru':
            net = GRUSMNIST(args.hidden).to(device)
            g = critical_gru_init(net, g=args.gain, scheme="gaussian")  #init_gain
        else:
            net = GGRUSMNIST(args.hidden, k=args.k).to(device)
            g = critical_ggru_init(net, g_star=args.gain,
                                           k=args.k, scheme="gaussian") #init_gain
            check_equivariance(net)          # unit test – aborts if fails

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
    mean_LE = np.sort(np.mean(all_LE, axis=0))[::-1]

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