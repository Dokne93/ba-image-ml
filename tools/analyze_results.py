#!/usr/bin/env python3
"""
Bereinigt die Qualitäts-CSV, erzeugt Kennzahlen & Plots.

Beispiel:
  python tools/analyze_results.py \
    --in outputs/quality.csv \
    --out outputs/quality_clean.csv \
    --plots outputs/plots
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Spalten, die bereinigt werden ----
NUM_COLS = [
    "psnr", "ssim", "lpips", "mse", "mae", "fsim", "vif",
    "brisque_cmp", "niqe_cmp", "dists"
]

# ---- Hilfsfunktionen ----
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Alle relevanten Spalten zu numerisch konvertieren."""
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Ungültige Werte bereinigen und Duplikate mitteln."""
    df = df.copy()

    if "psnr" in df:        df.loc[(df["psnr"] < 0) | (df["psnr"] > 100), "psnr"] = np.nan
    if "ssim" in df:        df.loc[(df["ssim"] < 0) | (df["ssim"] > 1), "ssim"] = np.nan
    if "lpips" in df:       df.loc[(df["lpips"] < 0) | (df["lpips"] > 1.5), "lpips"] = np.nan
    if "mse" in df:         df.loc[df["mse"] < 0, "mse"] = np.nan
    if "mae" in df:         df.loc[df["mae"] < 0, "mae"] = np.nan
    if "fsim" in df:        df.loc[(df["fsim"] < 0) | (df["fsim"] > 1), "fsim"] = np.nan
    if "vif" in df:         df.loc[(df["vif"] < 0) | (df["vif"] > 1.5), "vif"] = np.nan
    if "brisque_cmp" in df: df.loc[(df["brisque_cmp"] < 0) | (df["brisque_cmp"] > 100), "brisque_cmp"] = np.nan
    if "niqe_cmp" in df:    df.loc[(df["niqe_cmp"] < 0) | (df["niqe_cmp"] > 100), "niqe_cmp"] = np.nan
    if "dists" in df:       df.loc[(df["dists"] < 0) | (df["dists"] > 1.5), "dists"] = np.nan

    # Duplikate mitteln
    if "image" in df:
        df = df.groupby("image", as_index=False, dropna=False).agg({
            c: "mean" for c in df.columns if c != "image"
        })
    return df

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Count, Mean, Std, Min, Quartile, Max je Spalte."""
    stats = {}
    for c in NUM_COLS:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        stats[c] = {
            "count": len(s),
            "mean":  s.mean() if not s.empty else np.nan,
            "std":   s.std(ddof=1) if len(s) > 1 else np.nan,
            "min":   s.min() if not s.empty else np.nan,
            "25%":   s.quantile(0.25) if not s.empty else np.nan,
            "50%":   s.quantile(0.5) if not s.empty else np.nan,
            "75%":   s.quantile(0.75) if not s.empty else np.nan,
            "max":   s.max() if not s.empty else np.nan,
        }
    return pd.DataFrame(stats).T

# ---- Plot-Funktionen ----
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_bars(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    more = [c for c in ["psnr","ssim","fsim","vif"] if c in df.columns]
    less = [c for c in ["lpips","mse","mae","brisque_cmp","niqe_cmp","dists"] if c in df.columns]

    if more:
        means = df[more].mean()
        ax = means.sort_values(ascending=False).plot(kind="bar", color="green")
        ax.set_title("Durchschnitt (↑ besser)")
        ax.set_ylabel("Wert")
        ax.figure.tight_layout()
        ax.figure.savefig(outdir / "avg_more_is_better.png", dpi=160)
        plt.close(ax.figure)

    if less:
        means = df[less].mean()
        ax = means.sort_values(ascending=True).plot(kind="bar", color="red")
        ax.set_title("Durchschnitt (↓ besser)")
        ax.set_ylabel("Wert")
        ax.figure.tight_layout()
        ax.figure.savefig(outdir / "avg_less_is_better.png", dpi=160)
        plt.close(ax.figure)

def plot_scatter(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    if {"ssim","lpips"}.issubset(df.columns):
        tmp = df[["ssim","lpips","image"]].dropna()
        if not tmp.empty:
            fig = plt.figure()
            plt.scatter(tmp["ssim"], tmp["lpips"], alpha=0.7)
            for _, row in tmp.iterrows():
                plt.annotate(Path(str(row["image"])).name,
                             (row["ssim"], row["lpips"]),
                             fontsize=7, alpha=0.6)
            plt.xlabel("SSIM (↑ besser)")
            plt.ylabel("LPIPS (↓ besser)")
            plt.title("SSIM vs LPIPS")
            fig.tight_layout()
            fig.savefig(outdir / "scatter_ssim_lpips.png", dpi=160)
            plt.close(fig)

# ---- Hauptfunktion ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Pfad zur Eingabe-CSV")
    ap.add_argument("--out", dest="out_csv", required=True, help="Pfad zur bereinigten CSV")
    ap.add_argument("--plots", dest="plot_dir", default="", help="Ordner für Plots")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_csv = Path(args.out_csv)
    plot_dir = Path(args.plot_dir) if args.plot_dir else None

    if not inp.exists():
        raise SystemExit(f"Eingabe-CSV nicht gefunden: {inp}")

    df = pd.read_csv(inp)
    df = coerce_numeric(df)
    df_clean = clean_values(df)

    # Speichern
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_csv, index=False)

    # Statistik
    summ = summary_table(df_clean)
    print(f"\n[OK] Bereinigt -> {out_csv}  (Zeilen: {len(df_clean)})\n")
    print("Statistik (bereinigt):")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(summ.round(4))

    # Plots
    if plot_dir:
        plot_bars(df_clean, plot_dir)
        plot_scatter(df_clean, plot_dir)
        print(f"\n[OK] Plots gespeichert in: {plot_dir.resolve()}")

if __name__ == "__main__":
    main()
