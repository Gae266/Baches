import cv2
from ultralytics import YOLO

def analizar_bache_yolo(ruta_imagen, modelo_path="runs/detect/bache_detector/weights/best.pt"):
    # Cargar modelo YOLO entrenado
    modelo = YOLO(modelo_path)

    # Realizar inferencia
    resultados = modelo(ruta_imagen)

    # Obtener resultados de la primera imagen
    result = resultados[0]

    if not result.boxes:
        print("No se detectó el bache.")
        return

    # Cargar imagen original
    imagen = cv2.imread(ruta_imagen)

    # Dibujar cada caja detectada en la imagen
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        clase = int(box.cls[0])

        # Dibuja rectángulo y etiqueta
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
        etiqueta = f"Bache ({conf*100:.1f}%)"
        cv2.putText(imagen, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Guardar imagen con resultados
    salida = "resultado_deteccion.jpg"
    cv2.imwrite(salida, imagen)
    print(f"Resultado guardado como '{salida}'")

# Cambia la ruta de la imagen aquí si es necesario
ruta_imagen = "ProyectoBaches/bache1.jpg"
analizar_bache_yolo(ruta_imagen)