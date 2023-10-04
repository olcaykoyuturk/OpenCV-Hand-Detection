import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:

    ret, frame = cam.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hlms = hands.process(frameRGB)
    height, width, channel = frame.shape

    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:
            for fingerNum, landmark in enumerate(handlandmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)
                cv2.putText(frame, str(fingerNum), (positionX, positionY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                
            mpDraw.draw_landmarks(frame, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
