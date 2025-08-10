import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------- paths ----------
INPUT_DIR  = Path(r"C:\Users\Admin\Uni\LatexOCR\images")       
OUTPUT_DIR = Path(r"C:\Users\Admin\Uni\LatexOCR\preprocessed_images")  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- config ----------
GAUSS_K      = 3     # Gaussian kernel size (odd)
GAUSS_SIGMA  = 0     # 0 lets OpenCV choose from kernel
ADAPT_BLOCK  = 21    # odd, neighborhood size (>=3, odd)
ADAPT_C      = 10    # subtractive constant (tunes threshold)
INVERT       = False # True -> white bg / black ink, False -> black bg / white ink
EXTS = {".png"}  # file types to process
# ----------------------------

def preprocess_one(img_bgr, target_h, target_w):
    # 1) grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) adaptive threshold
    thresh_type = cv2.THRESH_BINARY_INV if INVERT else cv2.THRESH_BINARY
    bin_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        ADAPT_BLOCK, ADAPT_C
    )

    # 3) resize keeping aspect ratio, then pad to target_h x target_w
    h, w = bin_adapt.shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if new_w > w or new_h > h:
     interp = cv2.INTER_LINEAR  # upscale → smoother strokes
    else:
     interp = cv2.INTER_AREA   # downscale → avoid aliasing
    resized = cv2.resize(bin_adapt, (new_w, new_h), interpolation=interp)

    # create background
    bg_val = 255 if INVERT else 0
    canvas = np.full((target_h, target_w), bg_val, dtype=np.uint8)

    # center paste
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    return canvas

# ---- 1) Scan to get average dimensions ----
files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in EXTS]
files.sort()

heights, widths = [], []
for src in tqdm(files, desc="Scanning sizes"):
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    h, w = img.shape
    heights.append(h)
    widths.append(w)

avg_h = int(round(np.mean(heights)))
avg_w = int(round(np.mean(widths)))
print(f"Average height: {avg_h}, Average width: {avg_w}")

# ---- 2) Preprocess all images to average size ----
bad = 0
for src in tqdm(files, desc="Preprocessing"):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        bad += 1
        continue

    out = preprocess_one(img, avg_h, avg_w)

    dst = OUTPUT_DIR / (src.stem + ".png")
    cv2.imwrite(str(dst), out)

print(f"✅ Done. Wrote {len(files)-bad} images to {OUTPUT_DIR}  |  Skipped {bad} unreadable files.")


