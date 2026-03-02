import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ----------------------------
# Load memes
# ----------------------------
def load_image(name):
    img = cv2.imread(name)
    if img is None:
        print(f"❌ ERROR: Could not load {name}")
        exit()
    return img

def resize(img, w=350):
    return cv2.resize(img, (w, int(img.shape[0] * (w / img.shape[1]))))

meme_ok = resize(load_image("memes/ok.jpeg"))
meme_smile = resize(load_image("memes/SmiLe copy.jpg"))
meme_plain = resize(load_image("memes/PlAiN.jpg"))
meme_air = resize(load_image("memes/air.jpg"))
meme_punch = resize(load_image("memes/punch.png"))

meme_laugh = resize(load_image("memes/laugh.jpg"))
meme_shock = resize(load_image("memes/shock.jpg"))
meme_thumb = resize(load_image("memes/thumbs.jpg"))
meme_peace = resize(load_image("memes/peace.jpg"))

# ----------------------------
# Utility
# ----------------------------
def distance(a, b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5

# ----------------------------
# Hand gestures
# ----------------------------
def is_ok_sign(lm):
    return distance(lm[4], lm[8]) / distance(lm[0], lm[9]) < 0.3

def is_hands_up(lm):
    return all(lm[f].y < lm[f-2].y for f in [8, 12, 16, 20])

def is_punch(lm):
    return sum(lm[f].y > lm[f-2].y for f in [8, 12, 16, 20]) >= 3

def is_thumb_up(lm):
    return lm[4].y < lm[3].y and all(lm[f].y > lm[f-2].y for f in [8,12,16,20])

def is_peace(lm):
    return (
        lm[8].y < lm[6].y and
        lm[12].y < lm[10].y and
        lm[16].y > lm[14].y and
        lm[20].y > lm[18].y
    )

# ----------------------------
# Face detection (IMPROVED)
# ----------------------------
def face_reaction(face_results):
    if not face_results.multi_face_landmarks:
        return "plain"

    face = face_results.multi_face_landmarks[0]

    left = face.landmark[61]
    right = face.landmark[291]
    top = face.landmark[13]
    bottom = face.landmark[14]

    eye_top = face.landmark[159]
    eye_bottom = face.landmark[145]

    face_top = face.landmark[10]
    face_bottom = face.landmark[152]

    face_height = np.linalg.norm([face_top.x - face_bottom.x, face_top.y - face_bottom.y])

    mouth_width = np.linalg.norm([left.x - right.x, left.y - right.y])
    mouth_height = np.linalg.norm([top.x - bottom.x, top.y - bottom.y])
    eye_open = np.linalg.norm([eye_top.x - eye_bottom.x, eye_top.y - eye_bottom.y])

    width_ratio = mouth_width / face_height
    height_ratio = mouth_height / face_height
    eye_ratio = eye_open / face_height

    # 🎯 Final tuned logic
    if height_ratio > 0.30 and eye_ratio < 0.18:
        return "laugh"   # 😂

    elif height_ratio > 0.28 and eye_ratio > 0.22:
        return "shock"   # 😮

    elif width_ratio > 0.45 and height_ratio < 0.22:
        return "smile"   # 😊

    else:
        return "plain"

# ----------------------------
# Smoothing + cooldown
# ----------------------------
GESTURE_HISTORY = deque(maxlen=9)
last_gesture = "plain"
last_switch_time = 0
COOLDOWN = 1.2

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)

print("😂 Laugh | 😮 Shock | 👍 Thumbs | ✌️ Peace | 🙌 Hands | 👌 OK | 🥊 Punch")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    gesture = "none"

    # ----------------------------
    # Hand priority
    # ----------------------------
    if hand_results.multi_hand_landmarks:
        lm = hand_results.multi_hand_landmarks[0].landmark

        if is_punch(lm):
            gesture = "punch"
        elif is_thumb_up(lm):
            gesture = "thumbs_up"
        elif is_peace(lm):
            gesture = "peace"
        elif is_ok_sign(lm):
            gesture = "ok"
        elif is_hands_up(lm):
            gesture = "hands_up"

    # ----------------------------
    # Face fallback
    # ----------------------------
    if gesture == "none":
        gesture = face_reaction(face_results)

    # ----------------------------
    # Strong smoothing
    # ----------------------------
    GESTURE_HISTORY.append(gesture)
    gesture, count = Counter(GESTURE_HISTORY).most_common(1)[0]

    if count < 4:
        gesture = last_gesture

    # ----------------------------
    # Cooldown
    # ----------------------------
    now = time.time()
    if gesture != last_gesture and now - last_switch_time > COOLDOWN:
        last_gesture = gesture
        last_switch_time = now

    gesture = last_gesture

    # ----------------------------
    # Meme mapping
    # ----------------------------
    meme_map = {
        "ok": meme_ok,
        "hands_up": meme_air,
        "punch": meme_punch,
        "smile": meme_smile,
        "plain": meme_plain,
        "laugh": meme_laugh,
        "shock": meme_shock,
        "thumbs_up": meme_thumb,
        "peace": meme_peace
    }

    selected_meme = meme_map.get(gesture, meme_plain)

    # ----------------------------
    # Overlay meme
    # ----------------------------
    h, w, _ = selected_meme.shape
    if 10 + h < frame.shape[0] and 10 + w < frame.shape[1]:
        # Position on right side
        x_offset = frame.shape[1] - w - 10   # right margin
        y_offset = 10                        # keep top margin

    if y_offset + h < frame.shape[0] and x_offset + w < frame.shape[1]:
        roi = frame[y_offset:y_offset+h, x_offset:x_offset+w]
        frame[y_offset:y_offset+h, x_offset:x_offset+w] = cv2.addWeighted(
        roi, 0.25, selected_meme, 0.75, 0
    )

    # ----------------------------
    # Draw hand
    # ----------------------------
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ----------------------------
    # Transparent face mesh
    # ----------------------------
    if face_results.multi_face_landmarks:
        overlay = frame.copy()

        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                overlay,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(255, 255, 255), thickness=1
                )
            )

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    # ----------------------------
    # Display
    # ----------------------------
    cv2.imshow("Meme Reactor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()