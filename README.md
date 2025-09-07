# BA Image ML

**Bildverbesserung** (Denoising + Super-Resolution) und **Qualitätsmessung** (PSNR, SSIM, MSE, MAE, FSIM, VIF, LPIPS, DISTS), optimiert für **Linux** und **VS Code**.

Dieses Projekt ist Teil der Bachelorarbeit (Improving 3D Reconstruction Quality through Machine Learning ) und untersucht den Einfluss von Machine Learning auf die Bildqualität und die Genauigkeit von 3D-Rekonstruktionen.

---

## 📦 Setup

```bash
# Conda-Umgebung erstellen
conda env create -f environment.yml
conda activate ba-image-ml

# Optional: Torch mit CUDA installieren (nur falls GPU verfügbar)
# Hinweis: In diesem Projekt wurde nur das CPU-Setup getestet, bitte offizielle PyTorch-Anleitung beachten:
# https://pytorch.org/get-started/locally/

# Weitere Abhängigkeiten installieren
pip install -r requirements.txt

```

## 🚀 1. Bildverbesserung durchführen
```bash
python -m src.cli \
  --input data/raw \
  --output data/processed/ba_jpg \
  --steps denoise esrgan sharpen \
  --model realesrnet-x4plus \
  --weights models/RealESRNet_x4plus.pth \
  --tile 64 \
  --max-side 1600 \
  --workers 1 \
  --threads 6 \
  --out-ext jpg \
  --keep-size \
  -v

```
### Ablauf:
- Liest die Originalbilder aus data/raw/
- Schritte: Rauschreduktion → Super-Resolution (4x) → Schärfen
- Schreibt die bearbeiteten Bilder nach data/processed/ba_jpg/

## 📊 2. Qualität messen
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m src.metrics \
  --ref data/raw \
  --cmp data/processed/ba_jpg \
  --metrics psnr ssim mse mae fsim vif lpips dists \
  --workers 6 --threads 1 \
  --out outputs/quality_all.csv -v
```
### Ablauf:
- Vergleicht die verbesserten Bilder mit den Originalen.
- Berechnet:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - PSNR (Signal-Rausch-Verhältnis)
  - SSIM (Strukturelle Ähnlichkeit)
  - FSIM (Feature Similarity Index)
  - VIF (Visual Information Fidelity)
  - LPIPS (Perzeptuelle Bildqualität)
  - DISTS (Deep Image Structure and Texture Similarity)
- Speichert Ergebnisse als CSV: outputs/quality_all.csv

## 🔎 3. Ergebnisse prüfen
- Verbesserte Bilder: data/processed/ba_jpg/
- Metriken: outputs/quality_all.csv

## ⚙️ Wichtige Optionen (Pipeline)
- --skip-existing → überspringt bereits vorhandene Ausgabedateien
- --out-ext jpg|png → Ausgabeformat erzwingen
- --workers N → parallele Prozesse (empfohlen: 4)
- --threads N → Threads je Prozess (empfohlen: 1)
- -v / -vv → mehr Log-Ausgaben (INFO / DEBUG)
### Erweiterte Optionen
- --dry-run → nur Ordner anlegen / prüfen, keine Bilder verarbeiten
- --log-file outputs/run.log → schreibt Logdatei
- --fail-fast → bricht bei erstem Fehler ab (z. B. in CI)

## 📐 Workflow-Diagramm
```
   +-------------+        +------------------+        +-----------------+
   |   RAW DATA  | -----> |  Image Pipeline  | -----> | PROCESSED DATA  |
   |  (data/raw) |        | (denoise, ...)   |        | (data/processed)|
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
                          |  (outputs/*.csv) |
                          +------------------+

```
## 🎓 Wissenschaftliche Motivation

- **Kernpipeline (denoise → esrgan → sharpen):**  
  Dieser Workflow kombiniert drei zentrale Schritte:  
  1. **Denoising** reduziert Bildrauschen und verbessert die Stabilität von Feature-Detektoren (z. B. SIFT, ORB) in der 3D-Pipeline.  
  2. **Super-Resolution (ESRGAN)** erzeugt im Vergleich zu klassischen Interpolationsmethoden (bilinear, bicubic) realistischere Texturen und feinere Details. Für 3D-Rekonstruktionen bedeutet das schärfere Kanten und genauere Punktwolken.  
  3. **Sharpening** verstärkt lokale Kantenstrukturen, was die Kantenerkennung und geometrische Genauigkeit weiter unterstützt.  

- **Qualitätsmetriken:**  
  - **MSE (Mean Squared Error):** misst die mittlere quadratische Abweichung zwischen Original und verbessertem Bild.  
  - **MAE (Mean Absolute Error):** bewertet die durchschnittliche absolute Abweichung, robuster gegen Ausreißer.  
  - **PSNR (Peak Signal-to-Noise Ratio):** objektive Messung der Bildqualität, sensitiv auf Pixelabweichungen.  
  - **SSIM (Structural Similarity Index):** bewertet wahrgenommene Ähnlichkeit unter Berücksichtigung von Struktur, Luminanz und Kontrast.  
  - **FSIM (Feature Similarity Index):** fokussiert auf menschlich relevante Merkmale wie Kanten und Phaseninformationen und eignet sich besonders für Detailgenauigkeit.  
  - **VIF (Visual Information Fidelity):** quantifiziert den Erhalt der visuell relevanten Information im Vergleich zum Referenzbild.  
  - **LPIPS (Learned Perceptual Image Patch Similarity):** nutzt neuronale Netzwerke, um wahrgenommene visuelle Unterschiede besser zu erfassen.  
  - **DISTS (Deep Image Structure and Texture Similarity):** kombiniert tiefe Merkmalsrepräsentationen für Struktur- und Texturähnlichkeit.  

Die Kombination dieser Bearbeitungsschritte und Metriken stellt sicher, dass sowohl klassische, strukturelle als auch moderne, wahrnehmungsorientierte Qualitätsmaße berücksichtigt werden. Damit entsteht eine fundierte wissenschaftliche Bewertung der Einflüsse von Machine Learning auf die Bildqualität und deren Relevanz für 3D-Rekonstruktionen.

## ✅ Workflow Zusammenfassung
1. Bilder vorbereiten: Rohdaten in data/raw/
2. Verbesserung: src.cli → bearbeitete Bilder in data/processed/.../
3. Messung: src.metrics → Metriken in outputs/*.csv
4. Analyse: CSV + Bilder vergleichen, Ergebnisse für BA dokumentieren

## 🤖 Hinweis zum Einsatz von KI
Für Debugging und punktuelle Fehlerbehebung wurde ChatGPT (Modelle GPT-4o, GPT-5) unterstützend eingesetzt.  
Alle Konzepte, Methodenentscheidungen und Implementierungen stammen vom Autor.  

## 📜 Lizenz
Open Source – Nutzung und Erweiterung für Forschungszwecke ausdrücklich erlaubt.