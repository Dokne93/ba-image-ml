import multiprocessing as mp
from src.pipeline import main as _main, _build_parser

def main():
    # Stabile Startmethode für VS Code / PyTorch / NumPy
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # bereits gesetzt, kann ich ignorieren
        pass

    ap = _build_parser()
    args = ap.parse_args()
    _main(args)

if __name__ == "__main__":
    main()
