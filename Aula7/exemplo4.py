from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO("yolo26n.pt")

# Caminho do vídeo
video_path = "video.mp4"

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fim do vídeo ou erro na leitura do frame.")
        break

    frame_count += 1

    # Faz a detecção no frame atual
    results = model(frame)

    # Pega o primeiro resultado
    result = results[0]

    print(f"\nFrame {frame_count}")

    # Percorre todas as caixas detectadas
    for i, box in enumerate(result.boxes):
        # Coordenadas da bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Classe detectada
        cls_id = int(box.cls[0].item())
        cls_name = result.names[cls_id]

        # Confiança
        conf = float(box.conf[0].item())

        print(f"Objeto {i}:")
        print(f"  Classe: {cls_name}")
        print(f"  Confiança: {conf:.4f}")
        print(f"  Caixa: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")

    # Desenha as detecções no frame
    annotated_frame = result.plot()

    # Mostra na tela
    cv2.imshow("Deteccao com YOLO", annotated_frame)

    # Sai ao apertar q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()