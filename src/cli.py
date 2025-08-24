# src/cli.py
from .pipeline import main as _main

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--steps', nargs='+', required=True)
    ap.add_argument('--model', default='realesrgan-x4plus')
    ap.add_argument('--weights', default=None)

    # NEU: Tiling / Half / Max-Seite
    ap.add_argument('--tile', type=int, default=0, help='Tile-Größe für ESRGAN (0=aus, z.B. 256)')
    ap.add_argument('--half', action='store_true', help='FP16, nur sinnvoll mit GPU; CPU ignoriert es idR.')
    ap.add_argument('--max-side', type=int, default=0,
                    help='Eingabebild vor ESRGAN so skalieren, dass lange Kante <= max-side ist (0=aus)')

    args = ap.parse_args()
    _main(args)

if __name__ == '__main__':
    main()
