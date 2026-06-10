from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit

import matplotlib.pyplot as plt

COLUMNS = [
    "date", "player_a", "result_a", "score", "player_b", "result_b",
    "race_a", "race_b", "version", "venue"
]

@dataclass
class Metrics:
    n: int
    accuracy: float
    log_loss: float
    brier: float
    ece: float

def read_matches(path: Path) -> pd.DataFrame:
    """Read no-header StarCraft CSV and standardize a binary outcome y=A wins"""
    df = pd.read_csv(path, names=COLUMNS, header=None)
    df = df.dropna(subset=["player_a", "player_b", "result_a", "result_b", "race_a", "race_b"])
    df = df[df["result_a"].isin(["[winner]", "[loser]"])]
    df = df[df["result_b"].isin(["[winner]", "[loser]"])]
    df = df[df["result_a"] != df["result_b"]].copy()
    df["y"] = (df["result_a"] == "[winner]").astype(np.float64)
    for c in ["player_a", "player_b", "race_a", "race_b", "version", "venue"]:
        df[c] = df[c].astype(str).str.strip()
    return df.reset_index(drop=True)

def make_player_index(train: pd.DataFrame, valid: pd.DataFrame) -> Dict[str, int]:
    players = pd.concat([train["player_a"], train["player_b"], valid["player_a"], valid["player_b"]]).unique()
    return {p: i for i, p in enumerate(sorted(players))}

def make_extra_feature_names(train: pd.DataFrame, use_race: bool, use_context: bool) -> List[str]:
    names: List[str] = []
    if use_race:
        matchups = sorted((train["race_a"] + "_vs_" + train["race_b"]).unique())
        names += ["race:" + m for m in matchups]
    if use_context:
        venues = sorted(train["venue"].dropna().unique())
        versions = sorted(train["version"].dropna().unique())
        # One-hot all levels; L2 prior handles identifiability
        names += ["venue:" + v for v in venues]
        names += ["version:" + v for v in versions]
    return names

def build_design(df: pd.DataFrame, player_id: Dict[str,int], extra_names: List[str]) -> Tuple[csr_matrix, np.ndarray]:
    """Sparse design matrix. First columns are player skill effects: +1 for A, -1 for B."""
    n = len(df)
    nplayers = len(player_id)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=np.int64)
    vals = np.empty(2 * n, dtype=np.float64)
    cols[0::2] = df["player_a"].map(player_id).to_numpy()
    cols[1::2] = df["player_b"].map(player_id).to_numpy()
    vals[0::2] = 1.0
    vals[1::2] = -1.0
    X_skill = csr_matrix((vals, (rows, cols)), shape=(n, nplayers))

    if not extra_names:
        return X_skill, df["y"].to_numpy(dtype=np.float64)

    extra_id = {name: i for i, name in enumerate(extra_names)}
    erows: List[int] = []
    ecols: List[int] = []
    evals: List[float] = []

    # Add only features known from training. Unseen validation levels become all-zero.
    for r, row in enumerate(df.itertuples(index=False)):
        matchup = f"race:{row.race_a}_vs_{row.race_b}"
        venue = f"venue:{row.venue}"
        version = f"version:{row.version}"
        for feat in (matchup, venue, version):
            j = extra_id.get(feat)
            if j is not None:
                erows.append(r); ecols.append(j); evals.append(1.0)

    X_extra = csr_matrix((evals, (erows, ecols)), shape=(n, len(extra_names)))
    return hstack([X_skill, X_extra], format="csr"), df["y"].to_numpy(dtype=np.float64)

def fit_map(
    X: csr_matrix,
    y: np.ndarray,
    nplayers: int,
    sigma_skill: float = 1.5,
    sigma_extra: float = 3.0,
    maxiter: int = 200) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Fit MAP via penalized logistic likelihood. Return params and diagonal posterior std"""
    n_params = X.shape[1]
    prior_prec = np.empty(n_params, dtype=np.float64)
    prior_prec[:nplayers] = 1.0 / (sigma_skill ** 2)
    prior_prec[nplayers:] = 1.0 / (sigma_extra ** 2)

    def objective(w: np.ndarray) -> Tuple[float, np.ndarray]:
        z = X @ w
        p = expit(z)
        # Negative log posterior = logistic NLL + Gaussian prior penalty
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        prior = 0.5 * np.sum(prior_prec * w * w)
        grad = X.T @ (p - y) + prior_prec * w
        return float(nll + prior), np.asarray(grad)
    
    t0 = time.perf_counter()
    result = minimize(
        fun = objective,
        x0=np.zeros(n_params, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-5},
    )
    fit_seconds = time.perf_counter() - t0

    w = result.x
    p = expit(X @ w)
    v = p* (1.0 - p)
    # Diagonal Laplace approximation: H_diag = sum_i v_i x_ij^2 + prior precision_j
    H_diag = np.asarray(X.multiply(X).T @ v).ravel() + prior_prec
    post_std = 1.0 / np.sqrt(np.maximum(H_diag, 1e-12))
    info = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "fit_seconds": fit_seconds,
    }
    return w, post_std, info

def evaluate(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Metrics:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return Metrics(n=len(y), accuracy=acc, log_loss=ll, brier=brier, ece=float(ece))

def baseline_metrics(train: pd.DataFrame, valid: pd.DataFrame) -> Tuple[Metrics, np.ndarray]:
    """Baseline: use player A's empirical training win rate, smoothed to 0.5 for unseen players."""
    wins = {}
    games = {}
    for row in train.itertuples(index=False):
        for p, y in [(row.player_a, row.y), (row.player_b, 1.0 - row.y)]:
            wins[p] = wins.get(p, 0.0) + y
            games[p] = games.get(p, 0.0) + 1.0
    probs = []
    alpha = 2.0
    for row in valid.itertuples(index=False):
        p = (wins.get(row.player_a, 0.0) + alpha * 0.5) / (games.get(row.player_a, 0.0) + alpha)
        probs.append(p)
    probs = np.array(probs)
    return evaluate(valid["y"].to_numpy(dtype=np.float64), probs), probs

def plot_calibration(y: np.ndarray, p: np.ndarray, out_path: Path, title: str) -> None:
    bins = np.linspace(0, 1, 11)
    xs, ys, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))
    plt.figure(figsize=(5.2, 4.0))
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.scatter(xs, ys, s=np.maximum(20, np.sqrt(counts)))
    plt.xlabel("Mean predicted P(A wins)")
    plt.ylabel("Observed frequency of A wins")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_experiment(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(5.5, 4.0))
    for label, group in df.groupby("model"):
        group = group.sort_values(x)
        plt.plot(group[x], group[y], marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def run_one(train: pd.DataFrame, valid: pd.DataFrame, use_race: bool, use_context: bool,
            sigma_skill: float, frac: float, seed: int, maxiter: int) -> dict:
    rng = np.random.default_rng(seed)
    if frac < 1.0:
        idx = rng.choice(len(train), size=max(1000, int(frac * len(train))), replace=False)
        tr = train.iloc[np.sort(idx)].reset_index(drop=True)
    else:
        tr = train
    player_id = make_player_index(tr, valid)
    extra_names = make_extra_feature_names(tr, use_race=use_race, use_context=use_context)
    X_train, y_train = build_design(tr, player_id, extra_names)
    X_valid, y_valid = build_design(valid, player_id, extra_names)
    w, post_std, info = fit_map(X_train, y_train, len(player_id), sigma_skill=sigma_skill, maxiter=maxiter)
    pred = expit(X_valid @ w)
    met = evaluate(y_valid, pred)
    model_name = "skill"
    if use_race and use_context:
        model_name = "skill+race+context"
    elif use_race:
        model_name = "skill+race"
    elif use_context:
        model_name = "skill+context"
    return {
        "model": model_name,
        "train_fraction": frac,
        "sigma_skill": sigma_skill,
        "n_train": int(len(tr)),
        "n_players": int(len(player_id)),
        "n_features": int(X_train.shape[1]),
        "accuracy": met.accuracy,
        "log_loss": met.log_loss,
        "brier": met.brier,
        "ece": met.ece,
        "fit_seconds": info["fit_seconds"],
        "iterations": info["iterations"],
        "success": info["success"],
        "params": w,
        "post_std": post_std,
        "player_id": player_id,
        "extra_names": extra_names,
        "valid_pred": pred
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("."))
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test instead of the full report experiment.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = read_matches(args.data_dir / "train.csv")
    valid = read_matches(args.data_dir / "valid.csv")
    print(f"Loaded train={train.shape}, valid={valid.shape}")

    base, base_pred = baseline_metrics(train, valid)
    rows = [{
        "model": "baseline_player_winrate", "train_fraction": 1.0, "sigma_skill": np.nan,
        "n_train": len(train), "n_players": int(pd.concat([train.player_a, train.player_b]).nunique()),
        "n_features": 0, "accuracy": base.accuracy, "log_loss": base.log_loss,
        "brier": base.brier, "ece": base.ece, "fit_seconds": 0.0, "iterations": 0, "success": True
    }]

    # Main ablations: model structure and data size
    if args.quick:
        train = train.sample(n=min(12000, len(train)), random_state=args.seed).reset_index(drop=True)
        valid = valid.sample(n=min(4000, len(valid)), random_state=args.seed).reset_index(drop=True)
        configs = [(False, False), (True, True)]
        fracs = [0.25, 1.00]
        print(f"Quick mode: using train={train.shape}, valid={valid.shape}")
    else:
        configs = [(False, False), (True, False), (True, True)]
        fracs = [0.05, 0.10, 0.25, 0.50, 1.00]
    best_full = None
    for use_race, use_context in configs:
        for frac in fracs:
            print(f"Fitting use_race={use_race} use_context={use_context} frac={frac}")
            res = run_one(train, valid, use_race, use_context, sigma_skill=1.5,
                          frac=frac, seed=args.seed, maxiter=args.maxiter)
            rows.append({k: v for k, v in res.items() if k not in ["params", "post_std", "player_id", "extra_names", "valid_pred"]})
            if frac == 1.0 and res["model"] == "skill+race+context":
                best_full = res
    
    # Prior sensitivity for final full model
    for sigma in ([1.0, 1.5, 3.0] if args.quick else [0.5, 1.0, 1.5, 3.0, 6.0]):
        print(f"Fitting prior sensitivity sigma_skill={sigma}")
        res = run_one(train, valid, True, True, sigma_skill=sigma,
                      frac=1.0, seed=args.seed, maxiter=args.maxiter)
        rows.append({k: v for k, v in res.items() if k not in ["params", "post_std", "player_id", "extra_names", "valid_pred"]})
    
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "metrics_summary.csv", index=False)
    print(summary[["model", "train_fraction", "sigma_skill", "accuracy", "log_loss", "brier", "ece", "fit_seconds"]])

    # Figures for report.
    learn_df = summary[(summary["model"].isin(["skill", "skill+race", "skill+race+context"])) &
                       (summary["sigma_skill"].fillna(1.5) == 1.5)]
    plot_experiment(learn_df, "train_fraction", "log_loss", args.out_dir / "learning_curve_logloss.png",
                    "Validation log loss vs. training set size", "Training fraction", "Log loss")
    plot_experiment(learn_df, "train_fraction", "accuracy", args.out_dir / "learning_curve_accuracy.png",
                    "Validation accuracy vs. training set size", "Training fraction", "Accuracy")
    prior_df = summary[(summary["model"] == "skill+race+context") & (summary["train_fraction"] == 1.0)]
    plt.figure(figsize=(5.5, 4.0))
    plt.plot(prior_df["sigma_skill"], prior_df["log_loss"], marker="o")
    plt.xscale("log")
    plt.xlabel("Skill prior std. dev. sigma")
    plt.ylabel("Validation log loss")
    plt.title("Prior strength sensitivity")
    plt.tight_layout()
    plt.savefig(args.out_dir / "prior_sensitivity_log_loss.png", dpi=200)
    plt.close()

    if best_full is None:
        best_full = run_one(train, valid, True, True, sigma_skill=1.5,
                            frac=1.0, seed=args.seed, maxiter=args.maxiter)
    y_valid = valid["y"].to_numpy(dtype=np.float64)
    plot_calibration(y_valid, best_full["valid_pred"], args.out_dir / "calibration_full_model.png",
                     "Calibration: Bayesian Bradley-Terry full model")
    
    # Top skills with uncertainty for report table
    player_id = best_full["player_id"]
    inv_players = np.array([None] * len(player_id), dtype=object)
    for name, idx in player_id.items():
        inv_players[idx] = name
    nplayers = len(player_id)
    skills = best_full["params"][:nplayers]
    std = best_full["post_std"][:nplayers]
    skill_table = pd.DataFrame({"player": inv_players, "skill_map": skills, "posterior_sd_diag": std})
    skill_table["lower_approx_95"] = skill_table["skill_map"] - 1.96 * skill_table["posterior_sd_diag"]
    skill_table["upper_approx_95"] = skill_table["skill_map"] + 1.96 * skill_table["posterior_sd_diag"]
    skill_table = skill_table.sort_values("skill_map", ascending=False)
    skill_table.head(30).to_csv(args.out_dir / "top_players_with_uncertainty.csv", index=False)

    metadata = {
        "n_train_raw": int(len(train)),
        "n_valid_raw": int(len(valid)),
        "model_description": "Bayesian Bradley-Terry PGM with Gaussian priors on latent player skills and optional race/context effects.",
        "generated_files": sorted([p.name for p in args.out_dir.iterdir()])
    }
    with open(args.out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Results saved to {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()