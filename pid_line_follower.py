"""
Project 4: PID Line Follower

A simulated robot drives along a wavy path using a PID controller for
steering. Unlike Project 2 (gesture -> fixed command), this closes a
real control loop: the robot continuously measures how far it's
drifted from the path (the "error") and corrects itself, the same way
real line-following robots and cruise control systems work.

Live tuning (click the PyBullet window first so it has keyboard focus):
  1 / 2   : decrease / increase Kp (proportional - "how hard do I turn
             right now based on current error")
  3 / 4   : decrease / increase Ki (integral - "how much do I correct
             for error that's been building up over time")
  5 / 6   : decrease / increase Kd (derivative - "how much do I damp
             based on how fast the error is changing")
  r       : reset the robot to the start and clear the error log
  q       : quit and show the error-over-time plot

Try this once it's running: set Ki and Kd to 0 and only tune Kp. Past
a certain value you'll see the robot start oscillating side to side
instead of settling - that's classic proportional-only overshoot, and
it's why real controllers need the I and D terms too.
"""

import math
import time

import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

# --- Path definition: a sine wave the robot will try to follow ---
PATH_AMPLITUDE = 2.0     # meters, how wide the wave swings
PATH_WAVELENGTH = 8.0    # meters, distance for one full wave cycle
PATH_LENGTH = 40.0       # meters, total track length


def desired_y(x):
    return PATH_AMPLITUDE * math.sin(2 * math.pi * x / PATH_WAVELENGTH)


def path_heading(x):
    """Slope of the path at x, as an angle - used for a smoother error signal."""
    dy_dx = PATH_AMPLITUDE * (2 * math.pi / PATH_WAVELENGTH) * math.cos(2 * math.pi * x / PATH_WAVELENGTH)
    return math.atan(dy_dx)


FORWARD_SPEED = 2.0        # m/s, constant forward speed
MAX_STEER = 3.0             # rad/s, clamp on PID output
INTEGRAL_CLAMP = 5.0        # prevents integral windup


def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    # Draw the target path as a visible line in the world
    prev = None
    for i in range(int(PATH_LENGTH * 4) + 1):
        x = i / 4.0
        y = desired_y(x)
        pt = [x, y, 0.02]
        if prev is not None:
            p.addUserDebugLine(prev, pt, lineColorRGB=[1, 0.8, 0], lineWidth=2)
        prev = pt

    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.1])
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.1], rgbaColor=[0.9, 0.3, 0.3, 1]
    )
    robot_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, 0.15],
    )
    return robot_id


def reset_robot(robot_id):
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.15], [0, 0, 0, 1])
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])


def compute_error(pos):
    """Lateral error: how far off the path (in y) the robot is at its current x,
    blended with the path's local heading so the error signal is smoother
    around curves instead of jumping around."""
    x, y = pos[0], pos[1]
    target_y = desired_y(x)
    heading = path_heading(x)
    # Rotate the raw y-error into the path's local frame for a cleaner signal
    error = (y - target_y) * math.cos(heading)
    return error


def apply_drive(robot_id, angular_vel):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    yaw = p.getEulerFromQuaternion(orn)[2]
    vx = FORWARD_SPEED * math.cos(yaw)
    vy = FORWARD_SPEED * math.sin(yaw)
    p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, angular_vel])


def check_keys(gains):
    keys = p.getKeyboardEvents()
    changed = False
    step = {"kp": 0.2, "ki": 0.02, "kd": 0.05}

    key_map = {
        ord("1"): ("kp", -step["kp"]), ord("2"): ("kp", step["kp"]),
        ord("3"): ("ki", -step["ki"]), ord("4"): ("ki", step["ki"]),
        ord("5"): ("kd", -step["kd"]), ord("6"): ("kd", step["kd"]),
    }

    for key, (gain, delta) in key_map.items():
        if key in keys and keys[key] & p.KEY_WAS_TRIGGERED:
            gains[gain] = max(0.0, gains[gain] + delta)
            changed = True

    reset_requested = ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED
    quit_requested = ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED

    return changed, reset_requested, quit_requested


def main():
    robot_id = setup_simulation()

    gains = {"kp": 2.0, "ki": 0.0, "kd": 0.5}
    integral = 0.0
    prev_error = 0.0
    prev_time = time.time()

    time_log = []
    error_log = []
    sim_start = time.time()

    print("PID Line Follower running. Click the sim window, then use 1-6 to tune, r to reset, q to quit.")
    print(f"Starting gains: {gains}")

    last_printed_gains = dict(gains)

    while True:
        changed, reset_requested, quit_requested = check_keys(gains)
        if changed and gains != last_printed_gains:
            print(f"Gains updated: Kp={gains['kp']:.2f}  Ki={gains['ki']:.2f}  Kd={gains['kd']:.2f}")
            last_printed_gains = dict(gains)

        if quit_requested:
            break

        if reset_requested:
            reset_robot(robot_id)
            integral = 0.0
            prev_error = 0.0
            time_log.clear()
            error_log.clear()
            sim_start = time.time()
            print("Reset. Error log cleared.")

        if not p.isConnected():
            break

        pos, _ = p.getBasePositionAndOrientation(robot_id)
        error = compute_error(pos)

        now = time.time()
        dt = max(now - prev_time, 1e-3)
        prev_time = now

        integral += error * dt
        integral = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral))
        derivative = (error - prev_error) / dt
        prev_error = error

        steer = -(gains["kp"] * error + gains["ki"] * integral + gains["kd"] * derivative)
        steer = max(-MAX_STEER, min(MAX_STEER, steer))

        apply_drive(robot_id, steer)
        p.stepSimulation()

        time_log.append(now - sim_start)
        error_log.append(error)

        if pos[0] > PATH_LENGTH:
            reset_robot(robot_id)
            print("Reached end of path, looping back to start.")

        time.sleep(1.0 / 240.0)

    p.disconnect()

    if len(error_log) > 5:
        plt.figure(figsize=(10, 4))
        plt.plot(time_log, error_log, color="tab:red")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(
            f"PID Line-Following Error over Time  "
            f"(Kp={gains['kp']:.2f}, Ki={gains['ki']:.2f}, Kd={gains['kd']:.2f})"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Lateral Error (m)")
        plt.tight_layout()
        plt.savefig("pid_error_plot.png")
        print("Saved error plot to pid_error_plot.png")
        plt.show()
    else:
        print("Not enough data to plot - the sim was closed too quickly.")


if __name__ == "__main__":
    main()
