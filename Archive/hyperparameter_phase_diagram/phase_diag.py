# ================================================================
#  gru_smnist_lyap_repro.py  –  exact SMNIST-GRU pipeline (Vogt+24)
# ================================================================
"""
python phase_diag.py --phase_diagram --device cuda
python phase_diag.py --zoom --device cuda
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
import torch, itertools
import pandas as pd
import csv, itertools, time
import scipy.optimize as opt
import statsmodels.api as sm          # for logistic fit
import matplotlib.pyplot as plt



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


# permutation_testing
def check_equivariance(cell, input_dim, hidden_size, trials=25, atol=1e-6, rtol=1e-5, device='cpu'):
    """
    Monte-Carlo assertion that the cell is permutation-equivariant:
        h' = σ(x, Πh)  ⇔  Π h' = σ(x, h)
    Runs 'trials' random permutations.  Raises AssertionError on failure.
    """
    cell.eval()                                              # no dropout, etc.
    dtype = next(cell.parameters()).dtype
    for _ in range(trials):
        x = torch.randn(input_dim,  device=device, dtype=dtype)
        h = torch.randn(hidden_size, device=device, dtype=dtype)
        Π = torch.randperm(hidden_size, device=device)     # random perm

        y1  = cell(x, h)             # output with original state
        y2  = cell(x, h[Π])          # output with permuted state
        assert torch.allclose(y1[Π], y2, atol=atol, rtol=rtol), \
            "Permutation-equivariance test failed!"


class SharedRowLinear(nn.Module):
    """
    Weight matrix with identical rows: W = 1_{H×1} · wᵀ.
    Equivariant to any permutation of the out_features dimension.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.row = nn.Parameter(torch.empty(in_features))
        nn.init.uniform_(self.row, -0.05, 0.05)
        self.out_features = out_features
        # create on-the-fly to avoid device / dtype mismatch
        if bias:
            b = torch.zeros(1, dtype=self.row.dtype, device=self.row.device)
            self.register_parameter('bias', nn.Parameter(b))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):                          # x: (B, in_features)
        s = x @ self.row                           # (B,)
        ones = torch.ones(self.out_features, device=x.device, dtype=x.dtype)
        y = s.unsqueeze(-1) * ones                 # broadcast to (B, out_features)
        return y + (self.bias if self.bias is not None else 0.0)


# === Permutation-equivariant GRU (G-GRU) =========================
class PermEquiGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        # input → hidden (untied, identical to GRU)
        self.W_ir = SharedRowLinear(input_size, hidden_size, bias=False)
        self.W_iz = SharedRowLinear(input_size, hidden_size, bias=False)
        self.W_in = SharedRowLinear(input_size, hidden_size, bias=False)
        # hidden → hidden  (α·I + β·11ᵀ  ⇒ just two learnables / gate)
        for gate in ["r", "z", "n"]:
            self.register_parameter(f"alpha_{gate}",
                                    nn.Parameter(torch.randn(1)))
            self.register_parameter(f"beta_{gate}",
                                    nn.Parameter(torch.randn(1)))

    # affine map αh + β1ᵀh
    def _affine(self, gate: str, h: torch.Tensor) -> torch.Tensor:
        α, β = getattr(self, f"alpha_{gate}"), getattr(self, f"beta_{gate}")
        return α * h + β * h.mean(dim=-1, keepdim=True)  # invariant to permutations

    def forward(self, x_t, h_prev):
        r = torch.sigmoid(self.W_ir(x_t) + self._affine("r", h_prev))
        z = torch.sigmoid(self.W_iz(x_t) + self._affine("z", h_prev))
        n = torch.tanh( self.W_in(x_t) + r * self._affine("n", h_prev) )
        h_t = (1 - z) * n + z * h_prev
        return h_t

class PermEquiGRU(nn.Module):
    """Drop-in replacement for nn.GRU(batch_first=True, one layer)."""
    def __init__(self, input_size, hidden_size, bias=False):
        super().__init__()
        self.cell = PermEquiGRUCell(input_size, hidden_size, bias)
    def forward(self, seq, h0=None):
        B, T, _ = seq.shape
        if h0 is None:
            h = seq.new_zeros(B, self.cell.hidden_size)
        else:
            h = h0.squeeze(0)
        outs = []
        for t in range(T):
            h = self.cell(seq[:, t], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h.unsqueeze(0)
# ================================================================


# Model
class GRUSMNIST(nn.Module):
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.gru    = PermEquiGRU(28, hidden, bias=False)#nn.GRU(28, hidden, batch_first=True, bias=False)   # 28 × 28 → 28-step row stream
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
'''
def init_edge_of_chaos(model: nn.Module) -> float:
    p = torch.empty(1).uniform_(0.1, 30).item()
    # 1 ⟶ small orthogonal weights
    nn.init.orthogonal_(model.gru.weight_ih_l0)        # input -> hidden
    nn.init.orthogonal_(model.gru.weight_hh_l0)        # hidden -> hidden
    nn.init.zeros_(model.gru.bias_ih_l0)               # keep IH biases 0

    H = model.hidden
    with torch.no_grad():                              # 2 -> edge-of-chaos bias
        model.gru.bias_hh_l0[0*H:1*H] = -p             # reset-gate r
        model.gru.bias_hh_l0[1*H:2*H] =  p             # update-gate z
        model.gru.bias_hh_l0[2*H:3*H].zero_()          # candidate-n
    return p
'''
def init_edge_of_chaos(model):
    p = torch.empty(1).uniform_(0.1, 30).item()

    # sample from U(-p,p) **first**
    for w in [model.gru.weight_ih_l0, model.gru.weight_hh_l0]:
        nn.init.uniform_(w, -p, p)

    # orthogonalise afterwards (preserves scale p in each block)
    nn.init.orthogonal_(model.gru.weight_ih_l0)
    nn.init.orthogonal_(model.gru.weight_hh_l0)

    # gate-bias trick
    H = model.hidden
    with torch.no_grad():
        model.gru.bias_hh_l0[0:H]  = -p     # reset
        model.gru.bias_hh_l0[H:2*H] = +p    # update
        model.gru.bias_hh_l0[2*H:]  = 0.0   # candidate

    nn.init.zeros_(model.gru.bias_ih_l0)    # stay at zero
    return p


def init_uniform(model: nn.Module, p_min=0.1, p_max=3.0):
    p = float(torch.round((torch.rand(1)*(p_max-p_min)+p_min) * 1e3) / 1e3)

    for mod in model.modules():
        # (a) input → hidden: single row vector
        if isinstance(mod, SharedRowLinear):
            nn.init.uniform_(mod.row, -p, p)

        # (b) hidden → hidden: α, β scalars
        '''
        if isinstance(mod, PermEquiGRUCell):
            for name in ["alpha_r","beta_r","alpha_z","beta_z",
                         "alpha_n","beta_n"]:
                nn.init.uniform_(getattr(mod, name), -p, p)
        '''
        if isinstance(mod, PermEquiGRUCell):
            g = p * (3.0 * mod.hidden_size) ** 0.5   # √(3/H)
            for name in ["alpha_r", "beta_r", "alpha_z", "beta_z",
                         "alpha_n", "beta_n"]:
                nn.init.uniform_(getattr(mod, name), -g, g)
    open_gru_gates(model, p)     # σ(β_z)≈0.9 , σ(β_r)≈0.1
    return p

def open_gru_gates(model, p):
    """
    Force reset-gate ~ 0.1 and update-gate ~ 0.9 at init.
    Call immediately after init_uniform(model, p, p).
    """
    for mod in model.modules():
        if isinstance(mod, PermEquiGRUCell):
            with torch.no_grad():
                mod.beta_r.fill_(-p)   # r ≈ σ(-p) ≲ 0.1
                mod.beta_z.fill_(+p)   # z ≈ σ(+p) ≳ 0.9


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

    cell = model.gru.cell
    cell.eval()  

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
    with torch.no_grad():                       # avoids saving graphs
        for t in iterator:
            J = gru_jacobian_autograd(cell, seq[t], h)  # (H,H)
            Q, R = torch.linalg.qr(J @ Q)
            eps = torch.finfo(R.dtype).tiny
            le_sum += torch.log(torch.clamp(R.diagonal().abs(), min=eps))
            h = cell(seq[t], h)
            steps += 1

    return (le_sum / steps).cpu()                       # (H,)


def make_le_driver(batch=15, seq_len=100, device='cpu', dtype=torch.float64):
    # zero-mean, unit-variance Gaussian
    return torch.randn(batch, seq_len, 28, device=device, dtype=dtype)


def analyse_plateaus(LE: np.ndarray, eps: float = 1e-4):
    """
    Given a sorted Lyapunov spectrum (desc), print plateau lengths.
    """
    from itertools import groupby
    d = np.diff(LE)                      # NumPy first difference :contentReference[oaicite:5]{index=5}
    flat = np.abs(d) < eps
    runs = [len(list(g))+1 for k, g in groupby(flat) if k]
    if runs:
        print(f'Plateaus at multiplicities: {runs}')
    else:
        print('No plateaus detected.')


def zoom_phase_diagram(widths=(64, 128),
                       p_grid=np.arange(0.5, 3.05, 0.1),
                       seeds=50,
                       device='cpu'):
    """
    1) sweeps densely over p
    2) computes mean / std λ₁
    3) fits logistic to P(λ₁>0) and returns p_star
    4) saves a publication-ready .png
    """
    records = []
    for H, p, s in itertools.product(widths, p_grid, range(seeds)):
        torch.manual_seed(s)
        net = GRUSMNIST(hidden=H).double().to(device)
        init_uniform(net, p, p)             # calls open_gru_gates inside
        seq = make_le_driver(batch=1, seq_len=600, device=device)[0]
        lam = lyap_spectrum(net, seq, warm=500, progress=True)[0].item()
        records.append((H, p, lam))

    df = pd.DataFrame(records, columns=['width', 'p', 'lam'])
    out = []
    fig, ax1 = plt.subplots(figsize=(6,3))

    for H, g in df.groupby('width'):
        stats = g.groupby('p').lam.agg(['mean', 'std', 'count'])
        frac  = g.groupby('p').lam.apply(lambda x: (x>0).mean())
        # logistic fit ------------------------------------------------------
        x, y = frac.index.values, frac.values
        def logistic(x, a, b): return 1/(1+np.exp(-(x-a)/b))
        popt, _ = opt.curve_fit(logistic, x, y, p0=[1.5, .2])
        p_star  = popt[0]
        out.append((H, p_star))

        # -- plotting
        ax1.errorbar(x, stats['mean'], yerr=stats['std'],
                     label=f'H={H}', fmt='o', capsize=2)
        ax2 = ax1.twinx()
        ax2.plot(x, frac, linestyle='--')
        ax2.plot(x, logistic(x, *popt), linestyle='-', alpha=.4)

    ax1.set_xlabel('initial scale  $p$')
    ax1.set_ylabel(r'$\langle\lambda_{\max}\rangle \pm \sigma$')
    ax2.set_ylabel(r'$\Pr(\lambda_{\max}>0)$')
    ax1.axhline(0, ls=':', color='k')
    fig.tight_layout()
    fig.savefig('zoom_phase_diagram.png', dpi=300)
    print("saved  zoom_phase_diagram.png")
    for H, p_star in out:
        print(f'Critical p★ for H={H}: {p_star:.3f}')



def run_phase_diagram(widths      = (64, 128),
                      p_grid      = np.linspace(0.5, 3.5, 30),
                      seeds       = range(20),
                      device      = 'cpu',
                      dtype       = torch.float64,
                      warm_steps  = 500,
                      eval_steps  = 100):
    """
    Grid-search over (hidden width, init scale p, random seed).
    Saves phase_diagram.csv with columns:
        width, p, seed, lambda_max, n_pos, lyap_dim
    No training – spectra are measured **before** optimisation begins.
    """
    rows   = []
    t0     = time.time()
    total  = len(widths) * len(p_grid) * len(seeds)
    print(f"→ Running {total} spectra …")
    for H, p, seed in itertools.product(widths, p_grid, seeds):
        torch.manual_seed(seed)
        net = GRUSMNIST(hidden=H).to(device).double()
        init_uniform(net, p_min=p, p_max=p)        # deterministic scale
        #open_gru_gates(net, p)
        check_equivariance(net.gru.cell, 28, H, device=device)

        # single i.i.d. driver sequence
        seq = make_le_driver(batch=1,
                             seq_len=warm_steps+eval_steps,
                             device=device,
                             dtype=dtype)[0]
        LE  = lyap_spectrum(net, seq,
                            warm=warm_steps,
                            progress=False)
        analyse_plateaus(np.sort(LE)[::-1])

        # Kaplan–Yorke (Lyapunov) dimension
        #   D_KY = j + S_j/|λ_{j+1}| with S_j = Σ_{i≤j} λ_i  > 0 >= Σ_{i≤j+1} λ_i
        POS_TOL = 1e-6                    # numerical zero
        is_pos  = LE > POS_TOL
        λ_max    = LE[0].item()
        n_pos    = int(is_pos.sum().item())

        cum = torch.cumsum(LE, 0)

        if not is_pos.any():              # all λ_i ≤ 0  ⇒ fixed point
            dim_L = 0.0
        elif is_pos.all():                # all λ_i ≥ 0  ⇒ fully unstable
            dim_L = float(len(LE))        # = H
        else:
            j = int((cum > 0).sum().item() - 1)
            # guard for j == H-1  (cum still positive afterwards)
            if j + 1 == len(LE):
                dim_L = float(len(LE))
            else:
                dim_L = j + cum[j] / abs(LE[j + 1])

        rows.append(dict(width=H, p=p, seed=seed,
                         lambda_max=λ_max, n_pos=n_pos,
                         lyap_dim=dim_L))
        print(f"H={H:3d}  p={p:5.3f}  seed={seed:2d}  "
              f"λ₁={λ_max:+.4f}  n⁺={n_pos:2d}  D_KY={dim_L:5.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("phase_diagram.csv", index=False)
    print(f"Saved phase_diagram.csv  ({len(df)} rows)  "
          f"in {time.time()-t0:.1f}s")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch',  type=int, default=128)
    ap.add_argument('--lr',     type=float, default=1e-2)
    ap.add_argument('--trials', type=int, default=200,
                    help='number of independent runs to average')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--phase_diagram', action='store_true',
                help='run the automated p- and seed-sweep, save CSV, exit')
    ap.add_argument('--zoom', action='store_true',
                help='dense p-sweep + logistic fit + figure, then exit')
    args = ap.parse_args()

    if args.phase_diagram:
        run_phase_diagram(device=args.device)
        return                      # done – skip the training section
    if args.zoom:
        zoom_phase_diagram(device=args.device)
        return                 # skip everything else

    device = torch.device(args.device)
    _, _, teL = get_loaders(args.batch)

    all_LE = []        # will collect one (H,) array per trial
    all_p  = []        # keep the p that was drawn each run

    for run in range(1, args.trials + 1):
        # build & init model
        net = GRUSMNIST(args.hidden).to(device)
        p   = init_uniform(net) #init_edge_of_chaos(net)          # bias trick
        #open_gru_gates(net, p)
        all_p.append(p)

        # ---------- quick equivariance sanity-check
        check_equivariance(net.gru.cell, input_dim=28,
                           hidden_size=args.hidden,
                           trials=30, device=device)
        print("permutation-equivariance stress-test passed.")

        print(f"\n=== Trial {run}/{args.trials}  (p = {p:.3f}) ===")
        # ------------------------------------------------------------------
        # (after training) prepare one mini-batch from the TEST loader
        '''
        imgs, _ = next(iter(teL))               # imgs.shape → (128,1,28,28)
        imgs = imgs.to(device).double()         # to same dtype/device as model

        LE_batch = []
        for img in tqdm(imgs, desc="Lyap (128 seqs)"):# loop over the 128 sequences
            seq = img.view(28, 28)              # (T=28 , D=28)
            LE  = lyap_spectrum(net, seq, warm=0, progress=False).cpu().numpy()
            LE_batch.append(LE)

        LE = np.mean(LE_batch, axis=0)          # average over 128 sequences
        '''
        # ------------------------------------------------------------------
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
        analyse_plateaus(np.sort(LE)[::-1])

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
    print(f"p values drawn: {', '.join(f'{x:.3f}' for x in all_p)}")

    # plot mean spectrum
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.5,3.2))
    plt.plot(range(1, len(mean_LE)+1), mean_LE,
             marker='o', markersize=3, lw=1.5, label='mean of 100 trials')
    plt.axhline(0, color='black', lw=.8, ls='--')
    plt.xlabel('Exponent index  $i$')
    plt.ylabel(r'$\bar{\lambda}_i$')
    plt.title('Mean Lyapunov spectrum (100 trials)')
    plt.tight_layout()
    plt.savefig('lyap_spectrum_mean.png', dpi=300)
    print("Saved  lyap_spectrum_mean.npy  and  lyap_spectrum_mean.png")

if __name__ == '__main__':
    main()