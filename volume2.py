import cv2
import time
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV
cap = cv2.VideoCapture(0)
pTime = 0

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

colourVol = (255, 0, 0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = hand_landmarks.landmark

            # Find the coordinates of the thumb and index finger
            thumb_x = int(landmarks[4].x * img.shape[1])
            thumb_y = int(landmarks[4].y * img.shape[0])
            index_x = int(landmarks[8].x * img.shape[1])
            index_y = int(landmarks[8].y * img.shape[0])

            # Calculate the distance between thumb and index finger
            length = np.hypot(index_x - thumb_x, index_y - thumb_y)

            # Interpolate the distance to control volume
            volBar = np.interp(length, [30, 200], [400, 150])
            volPer = np.interp(length, [30, 200], [0, 100])

            # Smooth the volume changes
            smoothness = 10
            volPer = smoothness * round(volPer / smoothness)

            # Check which fingers are up
            fingers = [int(landmarks[i].y * img.shape[0]) < int(landmarks[i - 2].y * img.shape[0]) for i in range(5, 21, 4)]

            # Ensure there are enough elements in the fingers list
            while len(fingers) < 5:
                fingers.append(0)

            # If pinky finger is down, change the volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                colourVol = (0, 255, 0)
            else:
                colourVol = (255, 0, 0)

            # Draw bounding box
            x, y, w, h = cv2.boundingRect(np.array([[thumb_x, thumb_y], [index_x, index_y]]))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw lines between landmarks
            for i in range(len(landmarks)):
                lm = landmarks[i]
                lm_x, lm_y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                cv2.circle(img, (lm_x, lm_y), 7, (255, 0, 0), cv2.FILLED)

    # Draw the volume control bar and percentage
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Display the image with volume control information
    cv2.imshow("Hand Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
