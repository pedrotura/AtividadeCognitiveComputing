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

    # R2 - identifica a forma geométrica de maior massa com um retângulo verde em output visual
    maxArea = 0
    maxShape = 0
    for i in range(0, len(blue_contornos)):
        area = cv2.contourArea(blue_contornos[i])

        if area > maxArea:
            maxArea = area
            maxShape = blue_contornos[i]
    
    for i in range(0, len(reddish_contornos)):
        area = cv2.contourArea(reddish_contornos[i])

        if area > maxArea:
            maxArea = area
            maxShape = reddish_contornos[i]

    full_img = cv2.bitwise_and(frame, frame, blue_rgb)

    x, y, w, h = cv2.boundingRect(maxShape)
    cv2.rectangle(full_img, (x - 2, y - 2), (x + w - 2, y + h - 2), (116, 172, 68), 8)

    full_img = cv2.bitwise_and(full_img, full_img, reddish_rgb)

    # Exibe resultado
    cv2.imshow("Feed", full_img)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()