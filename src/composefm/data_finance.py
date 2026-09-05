"""Finance case study: order-flow channels as interventions on a market microstructure state.

FRAMING (see docs/PROJECT_RULES.md Rule 0). This module does NOT define a finance-specific
model. It supplies the three objects the GENERAL operator needs, for one domain:

    latent state z   <- microstructure feature vector of a time window
    intervention p    <- an order-flow CHANNEL (side x size-class) being active
    exposure tau      <- executed quantity in that channel (normalised)

The composition question is then the same one the method asks everywhere: given the
generators of single channels, predict the joint effect of several channels acting at once,
at exposures outside the training range.

Why this domain is a real test rather than a convenient one: market impact is known to be
strongly SUB-ADDITIVE in executed quantity -- the "square-root law", impact ~ Q^0.5, so
executing 2Q moves the price less than twice as far as executing Q. That is precisely the
contraction the gate gamma(z, P) is designed to represent, and it is an independently
established stylised fact of the field, not something we fitted. If the operator is right,
gamma should recover a concave exposure response WITHOUT being told to.

Data: Binance public bulk aggTrades (no key, CC-licensed bulk endpoint). One symbol-day is
~885k trades. Fields used: price, quantity, microsecond timestamp, and the aggressor side
(derivable from `is_buyer_maker`: if the buyer was the maker, the aggressor was a seller).
"""
from __future__ import annotations
import zipfile, pathlib
import numpy as np
import pandas as pd

AGG_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "ts",
            "is_buyer_maker", "best_match"]


def load_agg_trades(path: str | pathlib.Path) -> pd.DataFrame:
    """Read a Binance daily aggTrades zip into a tidy frame with a signed-quantity column."""
    z = zipfile.ZipFile(path)
    df = pd.read_csv(z.open(z.namelist()[0]), names=AGG_COLS)
    df["ts"] = pd.to_datetime(df["ts"], unit="us")
    # is_buyer_maker True => buyer was passive => the AGGRESSOR was a seller
    df["sign"] = np.where(df["is_buyer_maker"], -1.0, 1.0)
    df["signed_qty"] = df["sign"] * df["qty"]
    df["logp"] = np.log(df["price"].to_numpy())
    return df.sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------- channels
def assign_channels(df: pd.DataFrame, size_q: float = 0.90) -> pd.DataFrame:
    """Label every trade with one of 4 order-flow channels (the 'interventions').

    channel 0: small sell    channel 1: large sell
    channel 2: small buy     channel 3: large buy

    "Large" is the top decile of trade size, computed once on the whole sample so the
    threshold is a property of the data and not of any particular window.
    """
    thr = df["qty"].quantile(size_q)
    large = (df["qty"] >= thr).to_numpy()
    buy = (df["sign"] > 0).to_numpy()
    df = df.copy()
    df["channel"] = (buy.astype(int) * 2 + large.astype(int)).astype(np.int8)
    df.attrs["size_threshold"] = float(thr)
    return df


N_CHANNELS = 4
CHANNEL_NAMES = ["small sell", "large sell", "small buy", "large buy"]


# ---------------------------------------------------------------- state features
def _window_features(g: pd.DataFrame) -> np.ndarray:
    """Microstructure state of one window. Deliberately generic, no channel information."""
    lp = g["logp"].to_numpy()
    q = g["qty"].to_numpy()
    s = g["sign"].to_numpy()
    n = len(lp)
    r = np.diff(lp) if n > 1 else np.array([0.0])
    dur = max((g["ts"].iloc[-1] - g["ts"].iloc[0]).total_seconds(), 1e-3)
    tot = q.sum() + 1e-12
    # realised volatility, trade intensity, flow imbalance, size dispersion,
    # sign autocorrelation (order-splitting signature), range
    rv = float(np.sqrt(np.sum(r ** 2)))
    inten = float(np.log1p(n / dur))
    imb = float((s * q).sum() / tot)
    disp = float(np.std(np.log(q + 1e-12)))
    ac = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if n > 2 and np.std(s) > 0 else 0.0
    rng = float(lp.max() - lp.min())
    return np.array([rv, inten, imb, disp, ac, rng], dtype=np.float64)


STATE_DIM = 6
STATE_NAMES = ["realised vol", "log intensity", "flow imbalance",
               "size dispersion", "sign autocorr", "log range"]


def build_windows(df: pd.DataFrame, window: str = "10s", horizon: int = 1):
    """Split the day into windows; return (pre-state, post-state, per-channel exposure).

    For each window i:
      z0_i  = state features of window i
      z1_i  = state features of window i+horizon      (the 'response')
      tau_i = executed quantity per channel WITHIN window i, normalised to the
              per-channel median so tau ~ 1 is a typical exposure.
    """
    df = df.copy()
    df["bin"] = df["ts"].dt.floor(window)
    bins = sorted(df["bin"].unique())
    feats, expo = {}, {}
    for b, g in df.groupby("bin", sort=True):
        if len(g) < 5:
            continue
        feats[b] = _window_features(g)
        e = np.zeros(N_CHANNELS)
        for ch, gg in g.groupby("channel"):
            e[int(ch)] = gg["qty"].sum()
        expo[b] = e
    keys = [b for b in bins if b in feats]
    idx = {b: i for i, b in enumerate(keys)}
    Z = np.stack([feats[b] for b in keys])
    E = np.stack([expo[b] for b in keys])
    # normalise each channel's exposure by its median ACTIVE value
    med = np.array([np.median(E[E[:, c] > 0, c]) if (E[:, c] > 0).any() else 1.0
                    for c in range(N_CHANNELS)])
    Tau = E / med
    pairs = [(idx[keys[i]], idx[keys[i + horizon]])
             for i in range(len(keys) - horizon)]
    i0 = np.array([p[0] for p in pairs]); i1 = np.array([p[1] for p in pairs])
    return dict(Z=Z, Tau=Tau, med=med, keys=keys, i0=i0, i1=i1,
                z0=Z[i0], z1=Z[i1], tau=Tau[i0])


def standardise(z0: np.ndarray, z1: np.ndarray):
    """Whiten on the PRE-state only, so the transform is not shaped by the response."""
    mu, sd = z0.mean(0), z0.std(0) + 1e-9
    return (z0 - mu) / sd, (z1 - mu) / sd, mu, sd
