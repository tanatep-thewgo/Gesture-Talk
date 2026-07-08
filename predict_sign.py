import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# =========================
# โหลดโมเดล
# =========================
model = joblib.load("sign_model.pkl")

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# =========================
# Buffer กันกระพริบ
# =========================
pred_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # =========================
    # สร้าง UI panel ด้านขวา
    # =========================
    panel = np.zeros((h, 300, 3), dtype=np.uint8)

    top3_text = ["-", "-", "-"]
    top3_prob = [0, 0, 0]

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # สร้าง feature
            # =========================
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # =========================
            # ทำนาย + %
            # =========================
            probs = model.predict_proba([features])[0]
            classes = model.classes_

            # Top 3
            top3_idx = np.argsort(probs)[-3:][::-1]

            for i in range(3):
                top3_text[i] = classes[top3_idx[i]]
                top3_prob[i] = probs[top3_idx[i]] * 100

            # ใช้ตัวแรกเป็นหลัก
            pred_buffer.append(top3_text[0])
            final_pred = max(set(pred_buffer), key=pred_buffer.count)

    # =========================
    # วาด UI Panel
    # =========================

    cv2.putText(panel, "AI Prediction", (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    for i in range(3):

        y = 100 + i * 80

        # ตัวอักษร
        cv2.putText(panel,
                    f"{top3_text[i]}",
                    (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,255,0),
                    2)

        # เปอร์เซ็นต์
        cv2.putText(panel,
                    f"{top3_prob[i]:.1f}%",
                    (150, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2)

        # progress bar
        bar_width = int(top3_prob[i] * 2)
        cv2.rectangle(panel,
                      (30, y+20),
                      (30 + bar_width, y+35),
                      (0,255,0),
                      -1)

    # =========================
    # รวม frame + panel
    # =========================
    combined = np.hstack((frame, panel))

    cv2.imshow("Thai Sign AI Demo", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()