"""
Project 1: CV Playground - Hand Tracker
Detects your hand via webcam, draws landmarks, and counts raised fingers.

This is the same code style you'll reuse later for gesture-controlled
robot commands (Project 2) - the finger count / hand position here
becomes "forward", "stop", "turn" signals down the line.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Landmark indices for fingertips and the joint below each one
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def count_fingers(hand_landmarks, handedness_label):
    """Return how many fingers are extended (0-5)."""
    landmarks = hand_landmarks.landmark
    fingers_up = 0

    # Thumb: compare x-position instead of y (thumb moves sideways)
    if handedness_label == "Right":
        if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_PIPS[0]].x:
            fingers_up += 1
    else:
        if landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_PIPS[0]].x:
            fingers_up += 1

    # Other four fingers: tip above pip joint (lower y = higher on screen)
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            fingers_up += 1

    return fingers_up


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check that no other app is using it.")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        print("Hand tracker running. Press 'q' to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural movement
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    label = handedness.classification[0].label
                    finger_count = count_fingers(hand_landmarks, label)

                    cv2.putText(
                        frame,
                        f"Fingers: {finger_count}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

            cv2.imshow("Hand Tracker - Project 1", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
