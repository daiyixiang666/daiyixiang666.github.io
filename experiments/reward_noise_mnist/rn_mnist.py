"""Reward noise in diffusion RL on MNIST: Flow-GRPO vs DiffusionNFT.

Subcommands
  base        train an unconditional rectified-flow model on MNIST
  classifier  train the oracle classifier and a deliberately weak reward classifier
  rl          fine-tune the base model toward one target digit under a chosen reward-noise model
  eval-base   class histogram of the base model (sanity check: roughly 10% per digit)

Convention: x_t = (1 - t) x0 + t eps, v = eps - x0, sampling integrates t: 1 -> 0.
Images live in [-1, 1].

Noise models (--noise)
  clean                 R = 1[oracle(x) == target]
  sym:FP,FN             flip clean reward: P(R=1|wrong)=FP, P(R=0|correct)=FN, independent of x
  confuse:C,RATE[,FN]   only digit C is confused: P(R=1|digit C)=RATE; optional FN on target
  weak                  R = 1[weak_classifier(x) == target]   (instance-dependent, exploitable)
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def mnist(train=True):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return datasets.MNIST(str(ROOT / 'data'), train=train, download=True, transform=tf)


# ============================================================================ models
class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = 1000 * t[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class Res(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(te)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Small UNet for 28x28 (about 2.5M parameters with ch=64)."""

    def __init__(self, ch=64):
        super().__init__()
        td = ch * 4
        self.temb = TimeEmb(ch)
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.d1, self.d2 = Res(ch, ch, td), Res(ch, 2 * ch, td)          # 28, 14
        self.d3 = Res(2 * ch, 2 * ch, td)                                 # 7
        self.mid = Res(2 * ch, 2 * ch, td)
        self.u3 = Res(4 * ch, 2 * ch, td)
        self.u2 = Res(4 * ch, ch, td)
        self.u1 = Res(2 * ch, ch, td)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(t)
        h0 = self.inp(x)
        h1 = self.d1(h0, te)                                   # 28
        h2 = self.d2(F.avg_pool2d(h1, 2), te)                  # 14
        h3 = self.d3(F.avg_pool2d(h2, 2), te)                  # 7
        m = self.mid(h3, te)
        u = self.u3(torch.cat([m, h3], 1), te)
        u = F.interpolate(u, scale_factor=2)
        u = self.u2(torch.cat([u, h2], 1), te)
        u = F.interpolate(u, scale_factor=2)
        u = self.u1(torch.cat([u, h1], 1), te)
        return self.out(u)


class CNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.f = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1), nn.ReLU(), nn.Conv2d(w, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(), nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(2 * w * 49, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def features(self, x):
        return self.f(x)

    def forward(self, x):
        return self.head(self.f(x))


class WeakMLP(nn.Module):
    """Deliberately weak reward model: tiny MLP on a few hundred labels, so its errors are
    instance-dependent and a generator can find images it misreads."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        return self.net(x)


# ============================================================================ sampling
def timesteps(n, dev):
    return torch.linspace(1, 0, n + 1, device=dev)


@torch.no_grad()
def sample_ode(model, n, steps, dev, bs=512):
    out = []
    for i in range(0, n, bs):
        x = torch.randn(min(bs, n - i), 1, 28, 28, device=dev)
        ts = timesteps(steps, dev)
        for k in range(steps):
            t, tn = ts[k], ts[k + 1]
            v = model(x, t.expand(x.shape[0]))
            x = x + (tn - t) * v
        out.append(x.clamp(-1, 1))
    return torch.cat(out)


def sde_mean_std(model, x, t, tn, eta):
    """Flow-GRPO SDE step (t -> tn, dt < 0). Returns mean, std, v.
    mean = x + [v + s^2/(2t) (x + (1-t) v)] dt,  std = s sqrt(-dt),  s = eta sqrt(t/(1-t))."""
    tt = t.clamp(max=0.95)
    s = eta * torch.sqrt(tt / (1 - tt))
    dt = tn - t
    v = model(x, t.expand(x.shape[0]))
    mean = x + (v + s ** 2 / (2 * tt) * (x + (1 - tt) * v)) * dt
    std = s * torch.sqrt(-dt)
    return mean, std, v


def gauss_logp(x, mean, std):
    return (-((x - mean) ** 2) / (2 * std ** 2) - torch.log(std) - 0.5 * math.log(2 * math.pi)).mean(dim=(1, 2, 3))


@torch.no_grad()
def sample_sde_traj(model, n, steps, eta, dev):
    """Returns per-step states, next states and old log-probs for Flow-GRPO."""
    ts = timesteps(steps, dev)
    x = torch.randn(n, 1, 28, 28, device=dev)
    xs, xns, lps = [], [], []
    for k in range(steps):
        t, tn = ts[k], ts[k + 1]
        if k == steps - 1:            # last step deterministic (std -> 0 at t=0)
            v = model(x, t.expand(n))
            xn = x + (tn - t) * v
        else:
            mean, std, _ = sde_mean_std(model, x, t, tn, eta)
            xn = mean + std * torch.randn_like(x)
            xs.append(x); xns.append(xn); lps.append(gauss_logp(xn, mean, std))
        x = xn
    return x.clamp(-1, 1), xs, xns, torch.stack(lps, 1)   # lps: [n, steps-1]


# ============================================================================ rewards
class Rewarder:
    def __init__(self, spec, target, oracle, weak, gen):
        self.spec, self.target, self.oracle, self.weak, self.gen = spec, target, oracle, weak, gen

    @torch.no_grad()
    def __call__(self, x):
        pred = self.oracle(x).argmax(1)
        correct = (pred == self.target).float()
        u = torch.rand(x.shape[0], device=x.device, generator=self.gen)
        kind, _, arg = self.spec.partition(':')
        if kind == 'clean':
            R = correct
        elif kind == 'sym':
            fp, fn = map(float, arg.split(','))
            q = torch.where(correct.bool(), torch.full_like(u, 1 - fn), torch.full_like(u, fp))
            R = (u < q).float()
        elif kind == 'confuse':
            parts = arg.split(',')
            c, rate = int(parts[0]), float(parts[1])
            fn = float(parts[2]) if len(parts) > 2 else 0.0
            q = torch.where(correct.bool(), torch.full_like(u, 1 - fn),
                            torch.where(pred == c, torch.full_like(u, rate), torch.zeros_like(u)))
            R = (u < q).float()
        elif kind == 'weak':
            R = (self.weak(x).argmax(1) == self.target).float()
        else:
            raise ValueError(self.spec)
        return R, pred


@torch.no_grad()
def evaluate(model, oracle, rewarder, target, steps, dev, n=1000):
    x = sample_ode(model, n, steps, dev)
    pred = oracle(x).argmax(1)
    hist = torch.bincount(pred, minlength=10).float() / n
    R, _ = rewarder(x)
    # diversity among target-class samples: mean pairwise L2 in oracle feature space
    feats = oracle.features(x[pred == target])
    if feats.shape[0] > 2:
        d = torch.cdist(feats, feats)
        k = feats.shape[0]
        div = (d.sum() / (k * (k - 1))).item()
    else:
        div = float('nan')
    return {'oracle_acc': hist[target].item(), 'hist': hist.tolist(), 'noisy_reward': R.mean().item(),
            'diversity': div, 'ink': ((x + 1) / 2).mean().item(),
            'blank_frac': (((x + 1) / 2).mean(dim=(1, 2, 3)) < 0.02).float().mean().item()}, x[:64]


def save_grid(x, path):
    from torchvision.utils import save_image
    save_image((x + 1) / 2, str(path), nrow=8)


# ============================================================================ commands
def cmd_base(a):
    dev = device()
    torch.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(mnist(True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    model = UNet(a.ch).to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    print('params', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
    step = 0
    for ep in range(a.epochs):
        for x0, _ in dl:
            x0 = x0.to(dev)
            eps = torch.randn_like(x0)
            t = torch.rand(x0.shape[0], device=dev)
            xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
            loss = F.mse_loss(model(xt, t), eps - x0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - a.ema)
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        print(f'epoch {ep} loss {loss.item():.4f}', flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    out = ROOT / 'ckpt'; out.mkdir(exist_ok=True)
    torch.save({'ch': a.ch, 'state': ema.state_dict()}, out / 'base.pt')
    save_grid(sample_ode(ema, 64, 20, dev), out / 'base_samples.png')
    print('saved', out / 'base.pt')


def train_cls(model, ds, epochs, dev, lr=1e-3, bs=128):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in dl:
            loss = F.cross_entropy(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def acc(model, ds, dev):
    dl = torch.utils.data.DataLoader(ds, batch_size=1000)
    c = n = 0
    for x, y in dl:
        c += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return c / n


def cmd_classifier(a):
    dev = device()
    torch.manual_seed(a.seed)
    tr, te = mnist(True), mnist(False)
    oracle = train_cls(CNN().to(dev), tr, a.epochs, dev)
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(len(tr), generator=g)[:a.weak_n]
    weak = train_cls(WeakMLP().to(dev), torch.utils.data.Subset(tr, idx.tolist()), a.weak_epochs, dev, bs=32)
    out = ROOT / 'ckpt'; out.mkdir(exist_ok=True)
    torch.save(oracle.state_dict(), out / 'oracle.pt')
    torch.save(weak.state_dict(), out / 'weak.pt')
    info = {'oracle_test_acc': acc(oracle, te, dev), 'weak_test_acc': acc(weak, te, dev), 'weak_n': a.weak_n}
    # weak model's confusion on the test set for the target digit: P(weak says 3 | true digit)
    dl = torch.utils.data.DataLoader(te, batch_size=1000)
    conf = torch.zeros(10, 10)
    with torch.no_grad():
        for x, y in dl:
            p = weak(x.to(dev)).argmax(1).cpu()
            for yi, pi in zip(y, p):
                conf[yi, pi] += 1
    info['weak_confusion_rowsnorm'] = (conf / conf.sum(1, keepdim=True)).tolist()
    (out / 'classifiers.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if 'confusion' not in k}))


def load_all(dev):
    ck = torch.load(ROOT / 'ckpt' / 'base.pt', map_location=dev)
    base = UNet(ck['ch']).to(dev); base.load_state_dict(ck['state'])
    oracle = CNN().to(dev); oracle.load_state_dict(torch.load(ROOT / 'ckpt' / 'oracle.pt', map_location=dev))
    weak = WeakMLP().to(dev); weak.load_state_dict(torch.load(ROOT / 'ckpt' / 'weak.pt', map_location=dev))
    for m in (oracle, weak):
        m.eval().requires_grad_(False)
    return base, oracle, weak


def cmd_eval_base(a):
    dev = device()
    base, oracle, weak = load_all(dev)
    x = sample_ode(base.eval(), 5000, a.steps, dev)
    h = torch.bincount(oracle(x).argmax(1), minlength=10).float() / len(x)
    hw = torch.bincount(weak(x).argmax(1), minlength=10).float() / len(x)
    print('oracle hist', [round(v, 3) for v in h.tolist()])
    print('weak   hist', [round(v, 3) for v in hw.tolist()])


def group_adv(R, G):
    Rg = R.view(-1, G)
    A = (Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-4)
    return A.view(-1)


def cmd_rl(a):
    dev = device()
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    base, oracle, weak = load_all(dev)
    model = copy.deepcopy(base).train()
    ref = copy.deepcopy(base).eval().requires_grad_(False)
    old = copy.deepcopy(base).eval().requires_grad_(False)      # NFT: slowly updated sampling policy
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    rewarder = Rewarder(a.noise, a.target, oracle, weak, gen)
    name = a.name or f"{a.algo}_{a.noise.replace(':', '-').replace(',', '_')}_s{a.seed}"
    out = ROOT / 'runs' / name; out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=1))
    log = open(out / 'log.jsonl', 'w')
    n = a.groups * a.G
    t0 = time.time()
    for it in range(a.iters + 1):
        if it % a.eval_every == 0:
            model.eval()
            m, grid = evaluate(model, oracle, rewarder, a.target, a.eval_steps, dev, n=a.eval_n)
            m.update(it=it, time=time.time() - t0)
            log.write(json.dumps(m) + '\n'); log.flush()
            if it % (a.eval_every * 5) == 0 or it == a.iters:
                save_grid(grid, out / f'samples_{it:05d}.png')
            print(f"[{name}] it {it} oracle {m['oracle_acc']:.3f} noisyR {m['noisy_reward']:.3f} "
                  f"div {m['diversity']:.2f}", flush=True)
            model.train()
        if it == a.iters:
            break

        if a.algo == 'flowgrpo':
            sampler = model.eval()
            x0, xs, xns, lp_old = sample_sde_traj(sampler, n, a.steps, a.eta, dev)
            model.train()
            R, _ = rewarder(x0)
            A = group_adv(R, a.G)
            ts = timesteps(a.steps, dev)
            K = len(xs)
            for _ in range(a.inner):
                perm = torch.randperm(n, device=dev)
                for mb in perm.split(a.mb):
                    opt.zero_grad(set_to_none=True)
                    for k in range(K):                 # accumulate over denoising steps
                        mean, std, _ = sde_mean_std(model, xs[k][mb], ts[k], ts[k + 1], a.eta)
                        lp = gauss_logp(xns[k][mb], mean, std)
                        ratio = torch.exp(lp - lp_old[mb, k])
                        Ab = A[mb]
                        pg = -torch.min(ratio * Ab, ratio.clamp(1 - a.clip, 1 + a.clip) * Ab).mean()
                        loss = pg
                        if a.beta > 0:
                            with torch.no_grad():
                                mref, _, _ = sde_mean_std(ref, xs[k][mb], ts[k], ts[k + 1], a.eta)
                            kl = ((mean - mref) ** 2).mean(dim=(1, 2, 3)) / (2 * std ** 2)
                            loss = loss + a.beta * kl.mean()
                        (loss / K).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

        elif a.algo == 'nft':
            # 1) collect clean samples with the old policy (any sampler; ODE here)
            x0 = sample_ode(old, n, a.steps, dev)
            R, _ = rewarder(x0)
            if a.nft_reward == 'raw':
                r = R                                   # binary reward used as optimality probability
            elif a.nft_reward == 'center':              # raw reward re-centred per group at 0.5, no std scaling
                Rg = R.view(-1, a.G)
                r = (Rg - Rg.mean(1, keepdim=True) + 0.5).clamp(0, 1).view(-1)
            else:                                       # DiffusionNFT default: normalised advantage -> [0, 1]
                A = group_adv(R, a.G)
                r = 0.5 + 0.5 * (A / a.nft_zmax).clamp(-1, 1)
            # 2) negative-aware flow-matching loss on the forward process
            for _ in range(a.inner):
                perm = torch.randperm(n, device=dev)
                for mb in perm.split(a.mb):
                    xb, rb = x0[mb], r[mb]
                    t = torch.rand(len(mb), device=dev) * 0.98 + 0.01
                    eps = torch.randn_like(xb)
                    tb = t[:, None, None, None]
                    xt = (1 - tb) * xb + tb * eps
                    target = eps - xb
                    with torch.no_grad():
                        v_old = old(xt, t)
                    v = model(xt, t)
                    v_pos = (1 - a.nft_beta) * v_old + a.nft_beta * v
                    v_neg = (1 + a.nft_beta) * v_old - a.nft_beta * v
                    lpos = ((v_pos - target) ** 2).mean(dim=(1, 2, 3))
                    lneg = ((v_neg - target) ** 2).mean(dim=(1, 2, 3))
                    loss = (rb * lpos + a.nft_neg_weight * (1 - rb) * lneg).mean()
                    if a.beta > 0:                      # KL-like anchor to the reference velocity
                        with torch.no_grad():
                            v_ref = ref(xt, t)
                        loss = loss + a.beta * ((v - v_ref) ** 2).mean()
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            # 3) soft update of the sampling policy
            with torch.no_grad():
                for po, pm in zip(old.parameters(), model.parameters()):
                    po.lerp_(pm, 1 - a.nft_old_ema)
        else:
            raise ValueError(a.algo)
    log.close()
    torch.save(model.state_dict(), out / 'final.pt')


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('base')
    b.add_argument('--epochs', type=int, default=30); b.add_argument('--bs', type=int, default=256)
    b.add_argument('--lr', type=float, default=2e-4); b.add_argument('--ch', type=int, default=64)
    b.add_argument('--ema', type=float, default=0.999); b.add_argument('--seed', type=int, default=0)
    b.add_argument('--workers', type=int, default=0); b.add_argument('--max_steps', type=int, default=0)
    c = sp.add_parser('classifier')
    c.add_argument('--epochs', type=int, default=5); c.add_argument('--weak_n', type=int, default=300)
    c.add_argument('--weak_epochs', type=int, default=30); c.add_argument('--seed', type=int, default=0)
    e = sp.add_parser('eval-base'); e.add_argument('--steps', type=int, default=20)
    r = sp.add_parser('rl')
    r.add_argument('--algo', choices=['flowgrpo', 'nft'], required=True)
    r.add_argument('--noise', default='clean'); r.add_argument('--target', type=int, default=3)
    r.add_argument('--iters', type=int, default=300); r.add_argument('--G', type=int, default=16)
    r.add_argument('--groups', type=int, default=8); r.add_argument('--mb', type=int, default=64)
    r.add_argument('--inner', type=int, default=1); r.add_argument('--steps', type=int, default=10)
    r.add_argument('--eta', type=float, default=0.7); r.add_argument('--clip', type=float, default=0.2)
    r.add_argument('--lr', type=float, default=1e-5); r.add_argument('--beta', type=float, default=0.0)
    r.add_argument('--nft_beta', type=float, default=1.0); r.add_argument('--nft_reward', choices=['raw', 'norm', 'center'], default='norm')
    r.add_argument('--nft_neg_weight', type=float, default=1.0)
    r.add_argument('--nft_zmax', type=float, default=2.0); r.add_argument('--nft_old_ema', type=float, default=0.9)
    r.add_argument('--eval_every', type=int, default=10); r.add_argument('--eval_n', type=int, default=1000)
    r.add_argument('--eval_steps', type=int, default=20)
    r.add_argument('--seed', type=int, default=0); r.add_argument('--name', default='')
    a = p.parse_args()
    {'base': cmd_base, 'classifier': cmd_classifier, 'eval-base': cmd_eval_base, 'rl': cmd_rl}[a.cmd](a)


if __name__ == '__main__':
    main()
