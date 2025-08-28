# BA Image ML

GPU-beschleunigte **Bildverbesserung** (Denoising + Super-Resolution) und **Qualitätsmessung** (PSNR, SSIM, LPIPS) – optimiert für **Linux** + **VS Code**.

Dieses Projekt dient als Grundlage für die Bachelorarbeit und zeigt den Einfluss von Machine Learning auf die Bildqualität und die Genauigkeit von 3D-Rekonstruktionen.

## 📦 Setup

```bash
# Umgebung erstellen
conda env create -f environment.yml
conda activate ba-image-ml

# Torch passend zu deiner CUDA-Version installieren (siehe https://pytorch.org/get-started/locally/)
# Beispiel (CUDA 12.1):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# weitere Abhängigkeiten
pip install -r requirements.txt
```

## 🚀 1. Bildverbesserung durchführen
```bash
python -m src.cli \
  --input data/raw \
  --output data/processed/x4_dn_esr \
  --steps denoise esrgan sharpen \
  --model realesrgan-x4plus \
  --workers 2 --threads 2 -v
```
## Ablauf:
- Liest die Originalbilder aus data/raw/
- Schritte: Rauschreduktion → Super-Resolution (4x) → Schärfen
- Schreibt die bearbeiteten Bilder nach data/processed/x4_dn_esr/

## 📊 2. Qualität messen
```bash
python -m src.metrics \
  --ref data/raw \
  --cmp data/processed/x4_dn_esr \
  --metrics psnr ssim lpips \
  --out outputs/quality_x4_dn_esr.csv
```
### Ablauf:
- Vergleicht die verbesserten Bilder mit den Originalen.
- Berechnet:
  - PSNR (Signal-Rausch-Verhältnis)
  - SSIM (Strukturelle Ähnlichkeit)
  - LPIPS (Perzeptuelle Bildqualität)
- Speichert Ergebnisse als CSV: outputs/quality_x4_dn_esr.csv

## 🔎 3. Ergebnisse prüfen
- Verbesserte Bilder: data/processed/x4_dn_esr/
- Metriken: outputs/quality_x4_dn_esr.csv

## ⚙️ Wichtige Optionen (Pipeline)
- --skip-existing → überspringt bereits vorhandene Ausgabedateien
- --out-ext jpg|png → Ausgabeformat erzwingen
- --dry-run → nur Ordner anlegen / prüfen, keine Bilder verarbeiten
- --workers N → Anzahl paralleler Prozesse (empfohlen: CPU-Kerne/2)
- --threads N → Threads je Prozess (1–2 bei GPU empfohlen)
- --log-file outputs/run.log → schreibt Logdatei
- -v / -vv → mehr Log-Ausgaben (INFO / DEBUG)
- --fail-fast → bricht bei erstem Fehler ab (z. B. in CI)

## 📂 Beispiel: Batch-Run mit Logging
```bash
python -m src.cli \
  --input data/raw \
  --output data/processed/x4_all \
  --steps denoise median bilateral esrgan sharpen autobright \
  --model realesrgan-x4plus \
  --out-ext png \
  --workers 4 --threads 1 \
  --skip-existing \
  --log-file outputs/run.log \
  -vv
  ```

## 📐 Workflow-Diagramm
```
   +-------------+        +------------------+        +-----------------+
   |   RAW DATA  | -----> |  Image Pipeline  | -----> | PROCESSED DATA  |
   |  (data/raw) |        | (denoise, SR, …) |        | (data/processed)|
   +-------------+        +------------------+        +-----------------+
                                  |
                                  v
                        +------------------+
                        |  Metrics Module  |
                        | (PSNR, SSIM, ...)|
                        +------------------+
                                  |
                                  v
                         +------------------+
                         |   CSV Results    |
                         | (outputs/*.csv)  |
                         +------------------+

```
🎓 Wissenschaftliche Motivation

- Super-Resolution (ESRGAN): ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) wird eingesetzt, da es im Vergleich zu klassischen Interpolationsmethoden (Bilinear, Bicubic) deutlich realistischere Texturen erzeugt und feine Details rekonstruieren kann. Für 3D-Rekonstruktion bedeutet das: schärfere Kanten und genauere Punktwolken.

- Denoising: Bildrauschen stört Feature-Detektoren (z. B. SIFT, ORB) in der 3D-Pipeline. Vorverarbeitetes, entrauschtes Bildmaterial liefert robustere Matches und stabilere Kameraparameter.

- Qualitätsmetriken: 
  - PSNR (Peak Signal-to-Noise Ratio): objektive Messung der Bildqualität, sensitiv auf Pixelabweichungen.

  - SSIM (Structural Similarity Index): bewertet wahrgenommene Ähnlichkeit unter Berücksichtigung von Struktur, Luminanz und Kontrast.

  - LPIPS (Learned Perceptual Image Patch Similarity): nutzt neuronale Netzwerke, um wahrgenommene visuelle Unterschiede besser zu erfassen. → Diese Kombination deckt pixelbasierte, strukturbezogene und perzeptuelle Aspekte ab.

Die Wahl dieser Methoden stellt sicher, dass sowohl klassische Metriken als auch moderne, wahrnehmungsorientierte Maße berücksichtigt werden, wichtig für eine fundierte wissenschaftliche Bewertung.

## 💡 VS Code Tipps
### Debug-Konfiguration (.vscode/launch.json)
```bash
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Pipeline",
      "type": "python",
      "request": "launch",
      "module": "src.cli",
      "args": [
        "--input", "data/raw",
        "--output", "data/processed/x4_dn_esr",
        "--steps", "denoise", "esrgan", "sharpen",
        "--model", "realesrgan-x4plus",
        "--workers", "2",
        "--threads", "2",
        "-v"
      ]
    }
  ]
}
```
### Task zum Bauen (.vscode/tasks.json)
```bash
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Metrics",
      "type": "shell",
      "command": "python -m src.metrics --ref data/raw --cmp data/processed/x4_dn_esr --metrics psnr ssim lpips --out outputs/quality_x4_dn_esr.csv"
    }
  ]
}
```
Damit kannst du die Pipeline oder die Metriken direkt per Klick in VS Code starten.

## ✅ Workflow Zusammenfassung
1. Bilder vorbereiten: Rohdaten in data/raw/
2. Verbesserung: src.cli → bearbeitete Bilder in data/processed/.../
3. Messung: src.metrics → Metriken in outputs/*.csv
4. Analyse: CSV + Bilder vergleichen, Ergebnisse für BA dokumentieren

## 📜 Lizenz

Open Source – Nutzung und Erweiterung für Forschungszwecke ausdrücklich erlaubt.