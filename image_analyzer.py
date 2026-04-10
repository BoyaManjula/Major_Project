import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def analyze_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return "GENERAL_HEALTH"

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    face_result = mp_face.process(rgb)
    hand_result = mp_hands.process(rgb)

    face_detected = face_result.multi_face_landmarks is not None
    hand_detected = hand_result.multi_hand_landmarks is not None

    if face_detected and hand_detected:

        face = face_result.multi_face_landmarks[0].landmark

        mouth_left = face[61]
        mouth_right = face[291]

        mouth_center = (
            int((mouth_left.x + mouth_right.x)/2 * w),
            int((mouth_left.y + mouth_right.y)/2 * h)
        )

        for hand in hand_result.multi_hand_landmarks:

            for point_id in [4,8,12,16,20]:   # thumb + fingers
                finger = hand.landmark[point_id]

                hand_point = (
                    int(finger.x * w),
                    int(finger.y * h)
                )

                if distance(mouth_center, hand_point) < 100:
                    return "COUGH"

    if face_detected:
        return "HEADACHE"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0,20,70])
    upper = np.array([20,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    if cv2.countNonZero(mask) > 90000:
        return "SKIN_RASH"

    return "GENERAL_HEALTH"