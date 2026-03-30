from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO("yolo26n.pt")  
# model = YOLO("path/to/best.pt")

# Faz a predição
results = model("https://ultralytics.com/images/bus.jpg")

# Para cada resultado, gera e salva a imagem anotada
for i, result in enumerate(results):
    annotated_image = result.plot()  # imagem com bounding boxes, labels e confiança
    output_path = f"resultado_{i}.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Imagem salva em: {output_path}")