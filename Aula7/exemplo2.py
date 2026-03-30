from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO("yolo26n.pt")
# model = YOLO("path/to/best.pt")

# Abre a webcam
cap = cv2.VideoCapture(0)

# Verifica se a webcam abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

while True:
    # Lê um frame da webcam
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar frame da webcam.")
        break

    # Faz a predição no frame atual
    results = model(frame)

    # Pega o primeiro resultado e desenha os bounding boxes
    annotated_frame = results[0].plot()

    # Mostra o frame anotado
    cv2.imshow("Deteccao ao vivo", annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()