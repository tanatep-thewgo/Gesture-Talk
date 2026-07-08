import cv2
import mediapipe as mp
import csv

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

# =========================
# รับ label
# =========================
label = input(
    "Enter label (g / t / s / p / h / b / r / w / d / f / l / y / m / n / o / not_sign): "
).strip().lower()

# valid labels
valid_labels = [
    "g","t","s","p","h","b","r","w",
    "d","f","l","y","m","n","o","not_sign"
]

if label not in valid_labels:
    print("❌ Invalid label")
    exit()

# =========================
# เปิดไฟล์ CSV
# =========================
with open("hand_sign_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                features = []

                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                writer.writerow(features + [label])

                count += 1
                print(f"Saved {count}: {label}")

        # =========================
        # แสดง label บนหน้าจอ
        # =========================
        cv2.putText(
            frame,
            f"Collecting: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()