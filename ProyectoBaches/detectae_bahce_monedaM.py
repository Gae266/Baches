import cv2
from ultralytics import YOLO

# Diámetro real de la moneda de referencia en cm (ajusta si es otra moneda)
DIAMETRO_REAL_MONEDA_CM = 2.8

# Rutas a los modelos entrenados
RUTA_MODELO_MONEDA = "C:/Users/gaell/Documents/Visual Studio 2017/Python/runs/detect/train/weights/best.pt"
RUTA_MODELO_BACHE = "C:/Users/gaell/Documents/Visual Studio 2017/Python/runs/detect/bache_detector/weights/best.pt"

def detectar_objetos(modelo_path, ruta_imagen):
    modelo = YOLO(modelo_path)
    resultados = modelo(ruta_imagen)
    return resultados[0]

def calcular_escala_moneda(moneda_box):
    x1, y1, x2, y2 = map(int, moneda_box.xyxy[0])
    diametro_px = max(x2 - x1, y2 - y1)
    escala = DIAMETRO_REAL_MONEDA_CM / diametro_px  # cm por píxel
    return escala, (x1, y1, x2, y2)

def obtener_bache_mas_grande(bache_boxes):
    mayor_area = 0
    bache_grande = None
    for box in bache_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > mayor_area:
            mayor_area = area
            bache_grande = box
    return bache_grande

def calcular_dimensiones_bache(bache_box, escala):
    x1, y1, x2, y2 = map(int, bache_box.xyxy[0])
    ancho_px = x2 - x1
    alto_px = y2 - y1
    ancho_cm = ancho_px * escala
    alto_cm = alto_px * escala
    return ancho_cm, alto_cm, (x1, y1, x2, y2)

def analizar_imagen(ruta_imagen):
    moneda_result = detectar_objetos(RUTA_MODELO_MONEDA, ruta_imagen)
    bache_result = detectar_objetos(RUTA_MODELO_BACHE, ruta_imagen)

    if not moneda_result.boxes:
        print("No se detectó ninguna moneda.")
        return

    if not bache_result.boxes:
        print("No se detectó ningún bache.")
        return

    imagen = cv2.imread(ruta_imagen)

    # Escala de referencia con la primera moneda detectada
    escala, moneda_coords = calcular_escala_moneda(moneda_result.boxes[0])
    x1, y1, x2, y2 = moneda_coords
    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(imagen, "Moneda", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Obtener el bache más grande
    bache_grande = obtener_bache_mas_grande(bache_result.boxes)
    if bache_grande:
        ancho_cm, alto_cm, bache_coords = calcular_dimensiones_bache(bache_grande, escala)
        x1, y1, x2, y2 = bache_coords
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
        etiqueta = f"Bache más grande: {ancho_cm:.1f}cm x {alto_cm:.1f}cm"
        cv2.putText(imagen, etiqueta, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        print(f"Bache más grande - Ancho: {ancho_cm:.2f} cm, Alto: {alto_cm:.2f} cm")
    else:
        print("No se encontró un bache válido.")

    salida = "resultado_bache_mas_grande.jpg"
    cv2.imwrite(salida, imagen)
    print(f"Resultado guardado como '{salida}'")

# Ejecutar
ruta_imagen = "ProyectoBaches/bache1.jpg"
analizar_imagen(ruta_imagen)
