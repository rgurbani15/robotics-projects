"""
Project 2: Gesture-Controlled Simulated Robot

Reuses the finger-counting logic from Project 1 (hand_tracker.py) and
maps it to driving commands for a simple robot in a PyBullet physics
simulation. This is the perception -> action loop: camera sees your
hand, code decides a command, physics engine moves the robot.

Gesture map:
  0 fingers (fist)   -> STOP
  1 finger            -> TURN LEFT
  2 fingers           -> TURN RIGHT
  5 fingers (open)    -> FORWARD
  anything else       -> STOP (safe default)
"""

import time

import cv2
import mediapipe as mp
import pybullet as p
import pybullet_data

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

FORWARD_SPEED = 1.5   # m/s
TURN_SPEED = 1.5       # rad/s


def count_fingers(hand_landmarks, handedness_label):
    landmarks = hand_landmarks.landmark
    fingers_up = 0

    if handedness_label == "Right":
        if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_PIPS[0]].x:
            fingers_up += 1
    else:
        if landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_PIPS[0]].x:
            fingers_up += 1

    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            fingers_up += 1

    return fingers_up


def fingers_to_command(finger_count):
    """Translate a finger count into a (linear_velocity, angular_velocity) command."""
    if finger_count == 5:
        return FORWARD_SPEED, 0.0, "FORWARD"
    elif finger_count == 1:
        return 0.0, TURN_SPEED, "TURN LEFT"
    elif finger_count == 2:
        return 0.0, -TURN_SPEED, "TURN RIGHT"
    else:
        return 0.0, 0.0, "STOP"


def setup_simulation():
    """Create the PyBullet world and a simple box-chassis robot."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    # Simple box chassis standing in for a differential-drive robot.
    # (Real wheel dynamics come in Project 4 - for now we're closing
    # the perception -> action loop, not modeling motors yet.)
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.1])
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.1], rgbaColor=[0.2, 0.6, 1.0, 1]
    )
    robot_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, 0.15],
    )
    return robot_id


def apply_command(robot_id, linear_vel, angular_vel):
    """Drive the robot by directly setting its base velocity."""
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    yaw = euler[2]

    # Convert robot-frame linear velocity into world-frame x/y velocity
    import math
    vx = linear_vel * math.cos(yaw)
    vy = linear_vel * math.sin(yaw)

    p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, angular_vel])


def main():
    # Open the webcam FIRST, before the PyBullet window exists.
    # cv2.VideoCapture(0) can take a while to initialize on Windows,
    # and if PyBullet's window is already open during that wait, it
    # looks frozen and people (understandably) close it - which then
    # crashes the simulation calls below. Doing the slow step first
    # avoids that trap. CAP_DSHOW is also just faster than the default
    # backend on most Windows webcams.
    print("Opening webcam (this can take a few seconds on Windows)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    print("Webcam ready. Starting simulation window...")

    robot_id = setup_simulation()

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        print("Gesture-controlled sim running. Press 'q' in the webcam window to quit.")
        print("Gestures: open palm = forward, 1 finger = left, 2 fingers = right, fist = stop")

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            command_label = "STOP"
            linear_vel, angular_vel = 0.0, 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    label = handedness.classification[0].label
                    finger_count = count_fingers(hand_landmarks, label)
                    linear_vel, angular_vel, command_label = fingers_to_command(finger_count)

            if not p.isConnected():
                print("Simulation window was closed. Exiting.")
                break

            apply_command(robot_id, linear_vel, angular_vel)
            p.stepSimulation()

            cv2.putText(
                frame, f"Command: {command_label}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3,
            )
            cv2.imshow("Gesture Control - Project 2", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(1.0 / 60.0)  # keep sim + camera roughly in sync

    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()


if __name__ == "__main__":
    main()
