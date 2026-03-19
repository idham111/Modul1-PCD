import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from collections import Counter

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")


def get_dominant_color(img, k=4):
    """Menghitung warna dominan menggunakan K-Means clustering."""
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label_counts = Counter(labels.flatten())
    dominant_label = label_counts.most_common(1)[0][0]
    dominant_bgr = centers[dominant_label].astype(int)
    # Konversi BGR ke RGB
    return int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0])


def truncate_array(arr, max_rows=5, max_cols=10):
    """Memotong array agar tidak terlalu panjang untuk ditampilkan."""
    truncated = []
    rows = arr[:max_rows]
    for row in rows:
        if len(row) > max_cols:
            truncated.append(row[:max_cols] + ["..."])
        else:
            truncated.append(row)
    if len(arr) > max_rows:
        truncated.append(["..."])
    return truncated


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    with open(file_path, "wb") as f:
        f.write(image_data)

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    # === Fitur lama: RGB array (sekarang di-truncate) ===
    rgb_array = {
        "R": truncate_array(r.tolist()),
        "G": truncate_array(g.tolist()),
        "B": truncate_array(b.tolist()),
    }

    # === Fitur baru 1: Resolusi gambar ===
    height, width = img.shape[:2]
    resolution = f"{width} x {height} piksel"

    # === Fitur baru 2: Ukuran file (KB) ===
    file_size_kb = round(len(image_data) / 1024, 2)

    # === Fitur baru 3: Rata-rata warna RGB ===
    avg_r = round(float(np.mean(r)), 2)
    avg_g = round(float(np.mean(g)), 2)
    avg_b = round(float(np.mean(b)), 2)
    avg_color = {"R": avg_r, "G": avg_g, "B": avg_b}

    # === Fitur baru 4: Warna dominan ===
    dom_r, dom_g, dom_b = get_dominant_color(img)
    dominant_color = {"R": dom_r, "G": dom_g, "B": dom_b}

    return templates.TemplateResponse("display.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "rgb_array": rgb_array,
        "resolution": resolution,
        "file_size_kb": file_size_kb,
        "avg_color": avg_color,
        "dominant_color": dominant_color,
    })
