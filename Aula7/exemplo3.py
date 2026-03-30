from ultralytics import YOLO

# Carrega o modelo
model = YOLO("yolo26n.pt")
# model = YOLO("path/to/best.pt")

# Faz a predição
results = model("https://ultralytics.com/images/bus.jpg")

# Percorre os resultados
for i, result in enumerate(results):
    print(f"\nResultado {i}")

    boxes = result.boxes

    # Percorre cada caixa detectada
    for j, box in enumerate(boxes):
        # Coordenadas no formato x1, y1, x2, y2
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Coordenadas no formato x_center, y_center, width, height
        xc, yc, w, h = box.xywh[0].tolist()

        # Classe detectada
        cls_id = int(box.cls[0].item())
        cls_name = result.names[cls_id]

        # Confiança - A probabilidade de que a detecção seja correta
        conf = float(box.conf[0].item())

        print(f"Caixa {j}:")
        print(f"  Classe: {cls_name} (id={cls_id})")    # Exibe o nome da classe e seu ID
        print(f"  Confiança: {conf:.4f}")
        print(f"  xyxy: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
        print(f"  xywh: xc={xc:.2f}, yc={yc:.2f}, w={w:.2f}, h={h:.2f}")