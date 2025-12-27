import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime

# ===================== CONSTANTS =====================
REAL_SCALE_LENGTH_M = 0.30  # 30 cm
RED_AREA_MIN = 200
RED_AREA_MAX = 4000
PIXEL_TOLERANCE = 0.15  # ±15%

# ===================== MEDIAPIPE =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===================== RED MARKER DETECTION =====================
def find_scale_markers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
           cv2.inRange(hsv, lower_red2, upper_red2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if RED_AREA_MIN < area < RED_AREA_MAX:
            x, y, w, h = cv2.boundingRect(c)
            centers.append((x + w // 2, y + h // 2))

    if len(centers) != 2:
        return None

    return centers

# ===================== REACTION CATEGORY =====================
def classify_reaction(t):
    if t < 0.18:
        return "F1 Driver"
    elif t < 0.25:
        return "Elite Athlete"
    elif t < 0.35:
        return "Average Human"
    else:
        return "Slow Reaction"

# ===================== MAIN =====================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not detected.")
        return

    print("Press 'S' to start | 'Q' to quit")

    start_time = None
    pixel_to_meter = None
    triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # -------- HAND DETECTION --------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        finger_y = None
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark[8]
            finger_y = int(lm.y * h)
            cv2.circle(frame, (int(lm.x * w), finger_y), 6, (255, 0, 0), -1)

        # -------- SCALE DETECTION --------
        markers = find_scale_markers(frame)

        if markers:
            (x1, y1), (x2, y2) = markers
            cv2.circle(frame, (x1, y1), 6, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 6, (0, 255, 0), -1)

            pixel_length = abs(y2 - y1)
            pixel_to_meter = REAL_SCALE_LENGTH_M / pixel_length

            scale_bottom_y = max(y1, y2)

            # -------- TRIGGER CONDITION --------
            if start_time and finger_y and not triggered:
                if scale_bottom_y >= finger_y:
                    reaction_time = time.time() - start_time
                    category = classify_reaction(reaction_time)

                    print(f"Reaction Time: {reaction_time:.3f}s | {category}")

                    with open("reaction_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"{reaction_time:.3f}",
                            category
                        ])

                    triggered = True

        cv2.imshow("Reaction Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Started...")
            start_time = time.time()
            triggered = False

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== RUN =====================
if __name__ == "__main__":
    main()