from ultralytics import YOLO
import cv2

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# Carrega o modelo
model = YOLO("yolo26n.pt")

# Faz a predição
results = model("acidente.png")

for i, result in enumerate(results):
    print(f"\nResultado {i}")

    boxes = result.boxes
    carros = []

    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())

        if cls_id == 2:  # apenas carros
            carros.append({
                "indice": j,
                "box": (x1, y1, x2, y2)
            })

    # --- Verificação de sobreposição ---
    print("\nVerificando sobreposição entre carros...")
    for a in range(len(carros)):
        for b in range(a + 1, len(carros)):
            iou = calcular_iou(carros[a]["box"], carros[b]["box"])

            if iou >= 0.10:
                print(f"Sobreposição >= 10% entre {carros[a]['indice']} e {carros[b]['indice']}")

    # --- NOVA PARTE: gerar imagem com bounding boxes ---
    annotated_img = result.plot()  # imagem com TODAS as detecções

    # salvar como PNG
    cv2.imwrite("imagem_plotada.png", annotated_img)

    print("\nImagem salva como imagem_plotada.png")