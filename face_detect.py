import cv2
import mediapipe.python.solutions.face_detection as mp

face_detection = mp.FaceDetection()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            xmax = int((bbox.xmin + bbox.width) * w)
            ymax = int((bbox.ymin + bbox.height) * h)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()