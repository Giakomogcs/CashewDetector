### pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


#pip uninstall torch ultralytics
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip install ultralytics



#python -m venv env
#.\env\Scripts\activate
#pip install torch torchvision ultralytics
#python -c "from ultralytics import YOLO; print('YOLO imported successfully')"



import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from base64 import b64encode
import logging
import os
import sys

# Desativar mensagens de log de nível INFO do YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Caminho do modelo treinado
model_path = "Train_CashewNut/runs/detect/train/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Modelo não encontrado em {model_path}. Verifique o caminho.")
    sys.exit(1)

PRE_TRAINED_MODEL = YOLO(model_path)

# Labels e cores
LABELS = ['bad_cashew','good_cashew']
COLORS = {'bad_cashew': (0, 0, 255), 'good_cashew': (0, 255, 0)}

# Contadores globais
accumulated_counts = {label: 0 for label in LABELS}
track_id_history = {label: set() for label in LABELS}

def count_labels(track_id, class_id):
    global track_id_history
    label = LABELS[int(class_id)]

    if track_id not in track_id_history[label]:
        track_id_history[label].add(track_id)
        accumulated_counts[label] += 1

def main():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao abrir a webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame.")
                break

            detections = PRE_TRAINED_MODEL.track(np.array(frame), persist=True)
            for det in detections:
                for d in det.boxes.data.tolist():
                    if len(d) < 7:
                        continue
                    x1, y1, x2, y2, track_id, score, class_id = d[:7]
                    if score < 0.75:
                        continue

                    count_labels(track_id, class_id)
                    class_label = LABELS[int(class_id)]
                    color = COLORS[class_label]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{class_label} ID: {int(track_id)}", 
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            label_text = '\n'.join([f'{label}: {count}' for label, count in accumulated_counts.items()])
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            rect_w = max(text_size[0] + 20, 150)
            rect_h = (text_size[1] + 10) * len(accumulated_counts) + 20
            cv2.rectangle(frame, (5, 5), (rect_w, rect_h), (50, 50, 50), -1)

            y0, dy = 20, 20
            for i, line in enumerate(label_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
