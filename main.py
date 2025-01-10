import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import time   
from hand_detection import *
from track import *
# from picamera2 import Picamera2

# cargamos los emojis
emoji1 = cv2.imread("emoji_corazon.png", cv2.IMREAD_UNCHANGED)  
emoji2 = cv2.imread("emoji_perro.png", cv2.IMREAD_UNCHANGED)  
emoji3 = cv2.imread("emoji_fiesta.png", cv2.IMREAD_UNCHANGED)  
emoji4 = cv2.imread("emoji_smile.png", cv2.IMREAD_UNCHANGED)
emoji5 = cv2.imread("emoji_fuego.png", cv2.IMREAD_UNCHANGED)  

# posibles combinaciones de estados de la mano
combinations = {
    ("mano abierta", "puño cerrado", "mano abierta", "puño cerrado"): emoji1,
    ("puño cerrado", "puño cerrado", "mano abierta", "mano abierta"): emoji2,
    ("mano abierta", "mano abierta", "mano abierta", "mano abierta"): emoji3,
    ("puño cerrado", "mano abierta", "mano abierta", "puño"): emoji4,
    ("puño cerrado", "mano abierta", "puño cerrado", "mano abierta"): emoji5,
}

# inicializamos mediapipe para la detección de las manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

hand_states = []  # almacena los estados de la mano
cont_frames = 0 
valid_combination = False  
current_emoji = None  # emoji de la combinacion
final_combination = ""  # almacena la combinacion final

# picam = Picamera2()
# cap = picam.start()
cap = cv2.VideoCapture(0) # inicializar camara

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
    
prev_time = 0

while True:
    # frame = picam.capture_array()
    
    # if frame is None:
    #     print("Error: No se pudo capturar el fotograma.")
    #     break

    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer un fotograma.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    canny_frame = frame.copy()
    cont_frames += 1
    
    # calculo de FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    frame = draw_text_with_pillow(frame, f"FPS: {int(fps)}", (50, 400), font_size=1, color=(255, 255, 0))
    
    if not valid_combination:
        frame = draw_text_with_pillow(
            frame, 
            f"Combinación introducida: {', '.join(hand_states)}", 
            (50, 150 - 50), 
            font_size=0.8, 
            color=(255, 255, 255)
        )

    # si la combinacion es valida, muestra el emoji siguiendo la mano
    if valid_combination:
        if result.multi_hand_landmarks:  
            for landmarks in result.multi_hand_landmarks:
                x, y, w, h = detect_hand_shape(landmarks, frame)
                roi, canny_frame = apply_canny_on_original(frame, x, y, w, h)
                is_circle, circles = detect_hough_circle(roi)

                # determinamos el estado de la mano
                if is_circle:
                    hand_state = "puño cerrado"
                else:
                    hand_state = "mano abierta"

                # dibujamos el cuadrado alrededor de la mano
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # mostramos el emoji que sigue la mano
                if current_emoji is not None:
                    frame = overlay_emoji(frame, current_emoji, x, y, w, h)

            # mostramos el mensaje de combinación válida
            frame = draw_text_with_pillow(frame, "Combinación válida detectada!", (50, 50), font_size=2, color=(0, 255, 0))
            
            # una vez validada, mostramos la combinación final 
            frame = draw_text_with_pillow(frame, f"Combinación final: {final_combination}", (50, 150 - 50), font_size=0.8, color=(255, 255, 255))

        cv2.imshow("Imagen Original con Rectangulo y Emoji", frame)
        cv2.imshow("Bordes Canny con circulos", canny_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            x, y, w, h = detect_hand_shape(landmarks, frame)
            roi, canny_frame = apply_canny_on_original(frame, x, y, w, h)
            is_circle, circles = detect_hough_circle(roi)

            # determinamos el estado de la mano
            if is_circle:
                hand_state = "puño cerrado"
            else:
                hand_state = "mano abierta"

            # dibujamos el cuadrado alrededor de la mano
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # si la lista no está completa, añadimos el estado de la mano detectado
            if cont_frames % 50 == 0: 
                if len(hand_states) < 4:
                    hand_states.append(hand_state)
                    print(f"Estado añadido: {hand_state}")

                if len(hand_states) == 4:
                    most_common_state = tuple(hand_states)
                    print(f"Lista de estados: {hand_states}")
                    if most_common_state in combinations:
                        current_emoji = combinations[most_common_state]  # el emoji de la combinacion es el asignado a esta
                        valid_combination = True  
                        final_combination = ", ".join(hand_states)  
                        print("Combinación válida detectada, mostrando emoji.")
                    else: 
                        hand_states = []  

                    hand_states = []  

            if valid_combination:
                frame = draw_text_with_pillow(frame, "Combinación válida detectada!", (50, 50), font_size=2, color=(0, 255, 0))
            
            cv2.imshow("Imagen Original con Rectangulo y Emoji", frame)
            cv2.imshow("Bordes Canny con circulos", canny_frame)
    else:
        frame = draw_text_with_pillow(frame, "No se detectó ninguna mano", (50, 50), font_size=2, color=(255, 0, 0))
        cv2.imshow("Imagen Original con Rectangulo y Emoji", frame)
        cv2.imshow("Bordes Canny con circulos", canny_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar cámara y cerrar ventanas
# picam.stop()
cap.release()
cv2.destroyAllWindows()