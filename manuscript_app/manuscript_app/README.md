# Manuscript Cleaner (website + backend)

A small website that explains the project in plain language and lets anyone **upload a stained or faded page and get a
clean black-and-white version back**. Behind it is a FastAPI backend that runs your trained U-Net (`best_unet_v2.pth`).

```
Browser (static/index.html, css, js)  ──►  FastAPI (app/main.py)  ──►  U-Net on your GPU (app/engine.py)
```

## Quick start (Windows, PowerShell)

1. Put this folder anywhere, for example `D:\mbrepo\manuscript_app`, and open a terminal in it.
2. Install the web libraries (PyTorch is already installed in your environment, **do not reinstall it**):
   ```
   pip install -r requirements.txt
   ```
3. Give the app your trained model, either by copying `D:\mbrepo\outputs_v2\best_unet_v2.pth` into the `models` folder, or do nothing: the app also looks in `..\outputs_v2` and `D:\mbrepo\outputs_v2`.
4. Create the sample pages (real test pages the model never saw during training):
   ```
   python tools\make_samples.py --outputs D:\mbrepo\outputs_v2 --count 6
   ```
5. Check that the model runs on your GPU (about one minute):
   ```
   python tools\check_model.py
   ```
6. Start the website: double-click **`run_app.bat`**, or run
   ```
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   and open **http://127.0.0.1:8000**. The green pill in the top bar shows when the model is ready and whether it is using the GPU.

### Optional: "read the text" (OCR) panel
Install the Tesseract program (https://github.com/UB-Mannheim/tesseract/wiki) and `pip install pytesseract`.
The app finds `C:\Program Files\Tesseract-OCR\tesseract.exe` by itself (or set `TESSERACT_CMD`). Without it the OCR checkbox is switched off.

## What the website contains
| Section | What it says |
|---|---|
| Home | One-sentence promise, before/after picture |
| The problem | Stains, yellowing and ink bleed, in everyday words |
| How it works | Five simple steps, why AI beats simple filters, a collapsible technical box |
| Results | Score cards and two charts, taken from `static/data/results.json` (F1 0.854 vs 0.682 / 0.571; reading mistakes 46% → 23%), plus honest limitations |
| Try it | Upload, drag-and-drop, paste, sample pages, before/after slider, simple-filter and local-filter comparison, expert answer for samples, ink statistics, download, optional OCR |
| Questions | Privacy, best pictures, limits, DIBCO |

To change the numbers shown in *Results*, edit `static/data/results.json` (no rebuild needed).

## Project layout
```
manuscript_app/
  app/main.py        web server and the /api/clean job
  app/engine.py      the U-Net (copied from the notebook) + GPU/CPU loading
  app/tiling.py      cuts a page into overlapping tiles and blends the answers
  app/imaging.py     image reading, Otsu/Sauvola filters, PNG encoding, OCR
  static/            the website (index.html, css/style.css, js/app.js, data/results.json, samples/)
  tools/make_samples.py   builds sample pages from your held-out test pages
  tools/check_model.py    quick GPU self-test
  models/            put best_unet_v2.pth here
  run_app.bat        start script
```

## Settings (environment variables, all optional)
| Name | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | auto | full path of the `.pth` file |
| `MAX_UPLOAD_MB` | 20 | largest accepted upload |
| `MAX_SIDE` | 3000 | pictures larger than this (longest side, in pixels) are shrunk before cleaning |
| `TESSERACT_CMD` | auto | path to `tesseract.exe` |

Example: `set MAX_SIDE=4000` then start the app.

## API (for developers)
* `GET  /api/health`  model/GPU/OCR status
* `GET  /api/results` the numbers shown on the Results section
* `GET  /api/samples` the sample pages
* `POST /api/clean`   multipart form: `file` (image), `ocr` (true/false). Returns the original preview, the AI result, the two filter results (as data URLs), ink statistics, notes, and optional OCR text. Nothing is saved on the server.

## Letting other people try it
* **Same Wi-Fi / network:** start with `--host 0.0.0.0`, allow the port in the Windows firewall, and share `http://<your-computer-IP>:8000`. There is **no login**, so only do this on a network you trust.
* **Over the internet:** host the app on a machine that has the model (ideally with a GPU) behind a normal web server with HTTPS and a size/rate limit. The code is a standard FastAPI app.

## Troubleshooting
* **"Model not loaded"** (red pill): copy `best_unet_v2.pth` into `models/` and restart.
* **"CUDA problem … using the CPU"**: restart the app; check `nvidia-smi`. The CPU works, only slower.
* **Slow on big pages:** lower `MAX_SIDE`, or use the GPU.
* **Port already in use:** change `--port 8000` to another number.

## How this was tested
The backend was tested with a stand-in model (tile cutting/blending, all error cases, static files), the page logic was tested
in a simulated browser (27 checks), and the sample builder on a mock dataset. The real U-Net weights were **not** available
in the test environment, so run `tools\check_model.py` once on your machine to confirm the model loads.
