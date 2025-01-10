import cv2
import numpy as np

def detect_hand_shape(landmarks, frame):
    '''
    Función para detectar la forma de la mano
    '''
    h, w, _ = frame.shape
    points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks.landmark]
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    x, y, w, h = cv2.boundingRect(hull)
    margin = 40
    x -= margin
    y -= margin
    w += margin * 2
    h += margin * 2
    x = max(0, x)
    y = max(0, y)
    w = min(frame.shape[1] - x, w)
    h = min(frame.shape[0] - y, h)
    return (x, y, w, h)

def detect_hough_circle(roi):
    '''
    Función para detectar círculos en la región de interés
    '''
    if len(roi.shape) == 3:  
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
    blurred_roi = cv2.GaussianBlur(roi, (7, 7), 2)
    circles = cv2.HoughCircles(blurred_roi, cv2.HOUGH_GRADIENT, dp=1.26, minDist=20, 
                            param1=50, param2=30, minRadius=60, maxRadius=66)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return True, circles
    
    return False, None

def apply_canny_on_original(frame, x, y, w, h):
    '''
    Función para aplicar Canny en la imagen original
    '''
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (11, 11), 3)
    edges = cv2.Canny(blurred_frame, 20, 175)
    roi = edges[y:y+h, x:x+w]  # aplicamos Canny solo en la región del cuadrado
    canny_frame = np.zeros_like(frame)
    canny_frame[y:y+h, x:x+w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    is_circle, circles = detect_hough_circle(roi) # detectamos círculos solo en la región del cuadrado (ROI)

    if is_circle:
        for circle in circles:
            center = (circle[0], circle[1])  
            radius = circle[2]  
            cv2.circle(canny_frame, (x + center[0], y + center[1]), radius, (255, 0, 0), 2)

    return roi, canny_frame