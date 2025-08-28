# src/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import os
import gc
import traceback
import logging
import signal
import sys
import multiprocessing as mp  # <-- NEU

import cv2
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# interne Imports
from src.utils import list_images, read_image, save_image, relative_to_root
from src.denoise import opencv_fastnlmeans, median_blur, bilateral_filter
from src.enhance import esrgan_superres, unsharp_mask, auto_brightness



# ---------------------------
# Konfiguration / Konstanten
# ---------------------------
ALLOWED_STEPS = {
    "denoise",
    "median",
    "bilateral",
    "esrgan",
    "sharpen",
    "autobright",
}

DEFAULT_MODEL = "realesrgan-x4plus"


# ---------------------------
# Thread-/BLAS-Limits je Prozess
# ---------------------------
def _set_thread_limits(n: int):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        import torch  # type: ignore
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(n)
    except Exception:
        pass
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass


# ---------------------------
# Utilities
# ---------------------------
def _resize_long_side(img, max_side: int):
    """Skaliert so, dass die lange Kante <= max_side (nur wenn nötig)."""
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _choose_out_path(
    in_path: Path,
    in_root: Path,
    out_root: Path,
    out_ext: Optional[str],
) -> Path:
    """
    Leitet den Ausgabepfad aus dem Eingabepfad ab (relativ zu in_root).
    out_ext: z. B. ".png" oder "jpg" (Punkt optional). None = Extension beibehalten.
    """
    rel = relative_to_root(in_path, in_root)
    if out_ext is None:
        return out_root / rel
    ext = out_ext if out_ext.startswith(".") else f".{out_ext}"
    return (out_root / rel).with_suffix(ext)


def _validate_steps(steps: List[str]) -> Tuple[List[str], List[str]]:
    valid, invalid = [], []
    for s in steps:
        s_norm = s.strip().lower()
        if s_norm in ALLOWED_STEPS:
            valid.append(s_norm)
        else:
            invalid.append(s)
    # Eindeutigkeit bei Steps bewahren (Reihenfolge respektieren)
    seen = set()
    deduped = []
    for s in valid:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped, invalid


# ---------------------------
# Kern-Bildverarbeitung
# ---------------------------
def process_image(
    img,
    steps: List[str],
    model: str,
    weights: Optional[str] = None,
    tile: int = 0,
    half: bool = False,
    max_side: int = 0,
    keep_size: bool = False,   # <--- NEU
):
    out = img
    orig_h, orig_w = img.shape[:2]

    for s in steps:
        if s == "denoise":
            out = opencv_fastnlmeans(out)
        elif s == "median":
            out = median_blur(out, ksize=3)
        elif s == "bilateral":
            out = bilateral_filter(out)
        elif s == "esrgan":
            out = _resize_long_side(out, max_side)
            out = esrgan_superres(
                out,
                model_name=model,
                weights=weights,
                tile=tile,
                half=half,
            )
        elif s == "sharpen":
            out = unsharp_mask(out, radius=1, amount=0.6)
        elif s == "autobright":
            out = auto_brightness(out)
        else:
            logging.warning("Unknown step ignored: %s", s)

    # <<--- hier am Ende wieder auf Originalgröße zurückskalieren
    if keep_size:
        out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

    return out



# ---------------------------
# Worker-Funktion für Parallelbetrieb
# ---------------------------
def _process_one(
    p_str: str,
    in_dir: str,
    out_dir: str,
    steps: List[str],
    model: str,
    weights: Optional[str],
    tile: int,
    half: bool,
    max_side: int,
    threads: int,
    out_ext: Optional[str],
    dry_run: bool,
):
    """Wird in Subprozessen ausgeführt – gibt (pfad, fehler|None) zurück."""
    from pathlib import Path as _Path
    _set_thread_limits(threads)

    p = _Path(p_str)
    in_dir_p = _Path(in_dir)
    out_dir_p = _Path(out_dir)
    out_path = _choose_out_path(p, in_dir_p, out_dir_p, out_ext)

    try:
        if dry_run:
            # Nur Zielpfad anlegen prüfen
            (out_path.parent).mkdir(parents=True, exist_ok=True)
            return (p_str, None)

        img = read_image(p)
        out = process_image(
            img,
            steps=steps,
            model=model,
            weights=weights,
            tile=tile,
            half=half,
            max_side=max_side,
        )
        (out_path.parent).mkdir(parents=True, exist_ok=True)
        save_image(out_path, out)
        return (p_str, None)
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return (p_str, f"{e}\n{tb}")
    finally:
        try:
            del img, out  # type: ignore[name-defined]
        except Exception:
            pass
        gc.collect()


# ---------------------------
# CLI
# ---------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Batch-Bild-Pipeline (Denoise/ESRGAN/Sharpen/AutoBrightness ...)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Eingabeordner")
    ap.add_argument("--output", required=True, help="Ausgabeordner")
    ap.add_argument(
        "--steps",
        nargs="+",
        required=True,
        help=f"Verarbeitungsschritte in Reihenfolge. Erlaubt: {', '.join(sorted(ALLOWED_STEPS))}",
    )

    # ESRGAN / Enhancement
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Modell für Super-Resolution")
    ap.add_argument("--weights", default=None, help="Pfad zu Gewichten (optional)")
    ap.add_argument("--tile", type=int, default=0, help="Tile-Größe für ESRGAN (0=aus, z. B. 256)")
    ap.add_argument("--half", action="store_true", help="FP16 (nur sinnvoll mit GPU)")
    ap.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Vor ESRGAN lange Kante auf <= max-side skalieren (0=aus)",
    )

    # I/O Optionen
    ap.add_argument(
        "--out-ext",
        default=None,
        help="Ausgabe-Extension erzwingen (z. B. jpg, png). None = Original-Extension beibehalten.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bereits vorhandene Ausgabedateien überspringen (nach abgeleitetem Zielpfad).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur planen/prüfen: keine Bilder lesen/schreiben, nur Verzeichnisstruktur vorbereiten.",
    )

    # Parallelisierung
    ap.add_argument("--workers", type=int, default=1, help="Anzahl Prozesse (1=seriell)")
    ap.add_argument("--threads", type=int, default=1, help="CPU-Threads je Prozess")

    # Logging / Verhalten
    ap.add_argument(
        "--log-file",
        default=None,
        help="Optionaler Pfad für Logdatei (INFO-Level). Ohne Angabe nur Console-Logs.",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Bei erstem Fehler abbrechen (nützlich in CI).",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Mehr Ausgaben (ein- oder zweimal -v).",
    )
    ap.add_argument("--keep-size", action="store_true",
                help="Am Ende wieder auf die Originalgröße zurückskalieren.")

    return ap


def _setup_logging(log_file: Optional[str], verbose: int):
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def _install_sigint_handler():
    # Sorgt dafür, dass KeyboardInterrupt sauber propagiert
    signal.signal(signal.SIGINT, signal.default_int_handler)


def main(args: Optional[argparse.Namespace] = None):
    _install_sigint_handler()

    # args entweder von cli.py übergeben oder hier parsen
    if args is None:
        ap = _build_parser()
        args = ap.parse_args()

    _setup_logging(args.log_file, args.verbose)

    in_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Steps validieren
    steps, invalid = _validate_steps([*args.steps])
    if invalid:
        raise SystemExit(
            f"Ungültige Steps: {', '.join(invalid)}. Erlaubt: {', '.join(sorted(ALLOWED_STEPS))}"
        )
    if not steps:
        raise SystemExit("Keine gültigen Steps angegeben.")

    # Bilder sammeln + deterministisch sortieren
    images = list_images(in_dir)
    images = sorted(images, key=lambda p: str(p).lower())

    # Optional: bestehende Ausgaben filtern
    if args.skip_existing or args.dry_run:
        filtered = []
        for p in images:
            out_p = _choose_out_path(p, in_dir, out_dir, args.out_ext)
            if args.skip_existing and out_p.exists():
                logging.info("Skip (exists): %s", out_p)
                continue
            filtered.append(p)
        images = filtered

    total = len(images)
    if total == 0:
        print(f"Keine zu verarbeitenden Dateien gefunden in {in_dir} (nach Filtern).")
        return

    workers = int(getattr(args, "workers", 1))
    threads = int(getattr(args, "threads", 1))

    ok, fail = 0, 0

    # Serielle Verarbeitung (robust, speicherschonend)
    if workers <= 1:
        _set_thread_limits(threads)
        try:
            for idx, p in enumerate(images, 1):
                out_path = _choose_out_path(p, in_dir, out_dir, args.out_ext)
                try:
                    if args.dry_run:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        logging.info("[%d/%d] plan: %s -> %s", idx, total, p, out_path)
                        ok += 1
                        continue

                    img = read_image(p)
                    out = process_image(
                        img,
                        steps=steps,
                        model=args.model,
                        weights=args.weights,
                        tile=getattr(args, "tile", 0),
                        half=getattr(args, "half", False),
                        max_side=getattr(args, "max_side", 0),
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(out_path, out)
                    logging.info("[%d/%d] ok: %s -> %s", idx, total, p, out_path)
                    ok += 1
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logging.error("[ERROR] %s: %s", p, e)
                    fail += 1
                    if args.fail_fast:
                        raise
                finally:
                    try:
                        del img, out  # type: ignore[name-defined]
                    except Exception:
                        pass
                    gc.collect()
        except KeyboardInterrupt:
            print("\nAbgebrochen durch Benutzer (SIGINT).")
    else:
        # Parallel (mehrere Prozesse)
        func = partial(
            _process_one,
            in_dir=str(in_dir),
            out_dir=str(out_dir),
            steps=steps,
            model=args.model,
            weights=args.weights,
            tile=getattr(args, "tile", 0),
            half=getattr(args, "half", False),
            max_side=getattr(args, "max_side", 0),
            threads=threads,
            out_ext=args.out_ext,
            dry_run=args.dry_run,
        )
        try:
            # ---- WICHTIG: expliziter 'spawn'-Kontext für Stabilität ----
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futures = {ex.submit(func, str(p)): p for p in images}
                for idx, fut in enumerate(as_completed(futures), 1):
                    p = futures[fut]
                    try:
                        path_str, err = fut.result()
                        if err is None:
                            ok += 1
                            logging.info("[%d/%d] ok: %s", idx, total, path_str)
                        else:
                            fail += 1
                            logging.error("[ERROR] %s: %s", path_str, err.strip())
                            if args.fail_fast:
                                raise RuntimeError(err)
                    except KeyboardInterrupt:
                        # Sofort abbrechen: laufende Futures canceln
                        raise
        except KeyboardInterrupt:
            print("\nAbgebrochen durch Benutzer (SIGINT).")
        except Exception as e:
            logging.error("Abbruch wegen Fehler: %s", e)
            if args.fail_fast:
                raise

    print(f"Processed {ok} images (failed: {fail}) -> {out_dir}")


if __name__ == "__main__":
    # global auf 'spawn' setzen (falls noch nicht geschehen)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
