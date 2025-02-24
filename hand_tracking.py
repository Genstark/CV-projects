import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphand = mp.solutions.hands
hands = mphand.Hands()
mpdraw = mp.solutions.drawing_utils

ptime = 0
ctime = 0
num = 0 

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgb)

    if result.multi_hand_landmarks:
        for lndmrk in result.multi_hand_landmarks:
            for id, lm in enumerate(lndmrk.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) 
                # print(cx,cy)

            mpdraw.draw_landmarks(img, lndmrk, mphand.HAND_CONNECTIONS)
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Hand Detection",img)

    key = cv2.waitKey(1) & 0XFF
    if key == ord("q") or cv2.getWindowProperty('Hand Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()