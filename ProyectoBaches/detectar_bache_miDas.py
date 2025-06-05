import cv2
import numpy as np
import torch
import sys
import os
from ultralytics import YOLO
from torchvision.transforms import Compose

# === Añadir ruta a MiDaS (ajustar según sea necesario) === pip install timm
sys.path.append("C:/Users/gaell/Desktop/Baches/Codigo/MiDaS")

# === Importar MiDaS ===
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# === Parámetros ===
DIAMETRO_REAL_MONEDA_CM = 2.8
RUTA_MODELO_MONEDA = "runs/detect/train/weights/best.pt"
RUTA_MODELO_BACHE = "runs/detect/bache_detector/weights/best.pt"
RUTA_MODELO_MIDAS = "MiDaS/weights/dpt_beit_base_384.pt"

# === Inicializar MiDaS ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = DPTDepthModel(path=RUTA_MODELO_MIDAS, backbone="beitb16_384", non_negative=True).to(device).eval()
midas_transform = Compose([
    Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.5]*3, std=[0.5]*3),
    PrepareForNet()
])

def detectar_objetos(modelo_path, ruta_imagen):
    modelo = YOLO(modelo_path)
    resultados = modelo(ruta_imagen)
    return resultados[0]

def calcular_escala_moneda(moneda_box):
    x1, y1, x2, y2 = map(int, moneda_box.xyxy[0])
    diametro_px = ((x2 - x1) + (y2 - y1)) / 2  # promedio de alto y ancho
    escala = DIAMETRO_REAL_MONEDA_CM / diametro_px
    return escala, (x1, y1, x2, y2)

def calcular_dimensiones_bache(bache_box, escala):
    x1, y1, x2, y2 = map(int, bache_box.xyxy[0])
    ancho_px = x2 - x1
    alto_px = y2 - y1
    return ancho_px * escala, alto_px * escala, (x1, y1, x2, y2)

def estimar_mapa_profundidad(imagen_bgr):
    try:
        input_img = midas_transform({"image": imagen_bgr})["image"]
        input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=imagen_bgr.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        return prediction.cpu().numpy()
    except Exception as e:
        print(f"Error estimando la profundidad: {e}")
        return np.zeros(imagen_bgr.shape[:2])

def profundidad_media_en_region(mapa_profundidad, box):
    x1, y1, x2, y2 = box
    region = mapa_profundidad[y1:y2, x1:x2]
    if region.size == 0:
        return 0
    return np.mean(region)

def analizar_imagen(ruta_imagen):
    moneda_result = detectar_objetos(RUTA_MODELO_MONEDA, ruta_imagen)
    bache_result = detectar_objetos(RUTA_MODELO_BACHE, ruta_imagen)

    if moneda_result.boxes is None or len(moneda_result.boxes) == 0:
        print("No se detectó ninguna moneda.")
        return
    if bache_result.boxes is None or len(bache_result.boxes) == 0:
        print("No se detectó ningún bache.")
        return

    imagen = cv2.imread(ruta_imagen)
    mapa_profundidad = estimar_mapa_profundidad(imagen)

    escala, moneda_coords = calcular_escala_moneda(moneda_result.boxes[0])
    x1, y1, x2, y2 = moneda_coords
    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(imagen, "Moneda", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Seleccionar el bache más grande
    bache_mayor = max(bache_result.boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))

    ancho_cm, alto_cm, bache_coords = calcular_dimensiones_bache(bache_mayor, escala)
    prof_rel = profundidad_media_en_region(mapa_profundidad, bache_coords)

    # Normalización robusta
    min_val = np.percentile(mapa_profundidad, 5)
    max_val = np.percentile(mapa_profundidad, 95)
    prof_norm = np.clip((prof_rel - min_val) / (max_val - min_val), 0, 1)
    profundidad_aproximada_cm = prof_norm * 10

    x1, y1, x2, y2 = bache_coords
    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
    etiqueta = f"Bache: {ancho_cm:.1f}x{alto_cm:.1f}cm, Prof: {profundidad_aproximada_cm:.2f} cm"
    cv2.putText(imagen, etiqueta, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    print(f"Bache más grande - Ancho: {ancho_cm:.2f} cm, Alto: {alto_cm:.2f} cm, Profundidad estimada: {profundidad_aproximada_cm:.2f} cm")

    # Guardar resultados
    cv2.imwrite("resultado_completo.jpg", imagen)
    print("Resultado guardado como 'resultado_completo.jpg'")

    depth_vis = (255 * (mapa_profundidad - mapa_profundidad.min()) / (mapa_profundidad.max() - mapa_profundidad.min())).astype("uint8")
    cv2.imwrite("mapa_profundidad.jpg", depth_vis)
    print("Mapa de profundidad guardado como 'mapa_profundidad.jpg'")

# === Ejecutar ===
if __name__ == "__main__":
    ruta_imagen = "ProyectoBaches/bache3.jpg"
    analizar_imagen(ruta_imagen)