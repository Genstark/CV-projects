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

    box = cv2.rectangle(img, (300, 250), (100, 100), (0,255, 0), 2)

    if result.multi_hand_landmarks:
        for lndmrk in result.multi_hand_landmarks:
            for id, lm in enumerate(lndmrk.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) 
                # print(cx,cy)               
                
                if cx in box and cy in box:
                    cv2.putText(img, "Hand enter in the box", (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                    
            
            mpdraw.draw_landmarks(img, lndmrk, mphand.HAND_CONNECTIONS)
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    
    cv2.imshow("camera",img)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()