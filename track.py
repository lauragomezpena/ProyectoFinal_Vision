import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_text_with_pillow(frame, text, position, font_size=1, color=(0, 255, 0)):
    '''
    Función para dibujar texto en la imagen
    '''
    pillow_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pillow_image)
    font = ImageFont.truetype("arial.ttf", font_size * 20)
    draw.text(position, text, font=font, fill=color)
    frame = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
    return frame

def overlay_emoji(background, emoji, x, y, w, h):
    '''
    Función para superponer el emoji en la imagen
    '''
    if emoji.shape[2] == 4:
        emoji_rgb = emoji[:, :, :3]
        emoji_alpha = emoji[:, :, 3]
        emoji_resized = cv2.resize(emoji_rgb, (w, h))
        alpha_resized = cv2.resize(emoji_alpha, (w, h))
        mask = alpha_resized / 255.0
        mask_inv = 1.0 - mask
        roi = background[y:y+h, x:x+w]
        for c in range(3):  # para los 3 canales de color (RGB)
            roi[:, :, c] = (mask * emoji_resized[:, :, c] + mask_inv * roi[:, :, c])
        background[y:y+h, x:x+w] = roi
    return background