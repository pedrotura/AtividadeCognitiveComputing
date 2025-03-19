import cv2
import numpy as np

cap = cv2.VideoCapture('q1/q1B.mp4')


while True:
    ret, frame = cap.read()

    # R0 - não executa
    if not ret:
        print("Erro na captura do frame.")
        break
    
    # R1 - detecta todas as formas geométricas por cor e produz saída visual demonstrando
    blue_lower_hsv = np.array([105, 77, 130])
    blue_upper_hsv = np.array([115, 130, 230])

    reddish_lower_hsv = np.array([0, 60, 130])
    reddish_upper_hsv = np.array([15, 255, 255])

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    blue_hsv = cv2.inRange(img_hsv, blue_lower_hsv, blue_upper_hsv)
    blue_rgb = cv2.cvtColor(blue_hsv, cv2.COLOR_GRAY2RGB)

    reddish_hsv = cv2.inRange(img_hsv, reddish_lower_hsv, reddish_upper_hsv)
    reddish_rgb = cv2.cvtColor(reddish_hsv, cv2.COLOR_GRAY2RGB)

    blue_contornos, _ = cv2.findContours(blue_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    reddish_contornos, _ = cv2.findContours(reddish_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blue_img = blue_rgb.copy()
    reddish_img = reddish_rgb.copy()

    full_mask = cv2.bitwise_or(blue_rgb, reddish_rgb)

    full_img = cv2.bitwise_and(frame, frame, full_mask)
    
    cv2.drawContours(full_img, blue_contornos, -1, [0, 255, 0], 3)
    cv2.drawContours(full_img, reddish_contornos, -1, [0, 255, 0], 3)

    # Exibe resultado
    cv2.imshow("Feed", full_img)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()