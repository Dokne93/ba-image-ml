
# BA Image ML

GPU-beschleunigte Bildverbesserung (Denoising + Super-Resolution) und Qualitätsmessung (PSNR, SSIM, LPIPS) – optimiert für Linux + VS Code.

## Setup
```bash
conda env create -f environment.yml
conda activate ba-image-ml
# Torch passend zu deiner CUDA-Version installieren (siehe pytorch.org)
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r requirements.txt
```
## 1. Bildverbesserung durchführen
```
python -m src.cli --input data/raw --output data/processed/x4_dn_esr \
  --steps denoise esrgan sharpen --model realesrgan-x4plus
```

Was passiert dabei:

Nimmt die Originalbilder aus data/raw/

Führt Rauschreduktion → Super-Resolution (4x) → leichtes Schärfen durch

Speichert die bearbeiteten Bilder in data/processed/x4_dn_esr/

## 2. Qualität messen
```
python -m src.metrics --ref data/raw --cmp data/processed/x4_dn_esr \
  --metrics psnr ssim lpips --out outputs/quality_x4_dn_esr.csv
```

Was passiert dabei:

Vergleicht die verbesserten Bilder mit den Originalen

Berechnet:

PSNR (Signal-Rausch-Verhältnis)

SSIM (Strukturelle Ähnlichkeit)

LPIPS (Perzeptuelle Bildqualität)

Speichert die Ergebnisse als CSV in outputs/quality_x4_dn_esr.csv

## 3. Ergebnisse prüfen

Die verbesserten Bilder sind unter:

data/processed/x4_dn_esr/


Die CSV mit den Qualitätswerten unter:

outputs/quality_x4_dn_esr.csv


