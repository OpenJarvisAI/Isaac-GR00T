import time
import threading
import json_numpy
import numpy as np
import requests
import queue
import cv2
import pickle, os
from matplotlib.pyplot import step
from typing import Any, cast

json_numpy.patch()

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

from loguru import logger

try:
    import pyrealsense2 as rs
except ImportError:
    print("Warning: pyrealsense2 not available. Camera functionality will be limited.")
    rs = None

# Help static analyzers: treat rs as dynamic Any when available
if rs is not None:
    rs = cast(Any, rs)


class CameraWrapper:
    def __init__(
        self, devices=None, width=640, height=480, fps=30, num_realsense=0, cv_format="MJPEG"
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.num_realsense = max(0, int(num_realsense))
        self.cv_format = cv_format
        self.cameras = []  # list of dicts: {type: 'rs'|'cv', handle: pipeline|cap}
        self.device_ids = devices if devices is not None else []
        self._open_cameras()
        print(f"successfully opened {len(self.cameras)} cameras!")

    def _open_cameras(self):
        if not self.device_ids:
            print("No devices provided for CameraWrapper")
            return

        for idx, dev in enumerate(self.device_ids):
            # Decide camera type
            use_realsense = idx < self.num_realsense

            if use_realsense:
                if rs is None:
                    print(
                        f"pyrealsense2 not available, skipping RealSense device at index {idx} (id: {dev})"
                    )
                    continue
                try:
                    serial = str(dev)
                    pipeline = rs.pipeline()  # type: ignore[attr-defined]
                    config = rs.config()  # type: ignore[attr-defined]
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # type: ignore[attr-defined]
                    pipeline.start(config)
                    self.cameras.append({"type": "rs", "handle": pipeline})
                    print(f"RealSense camera {serial} opened successfully")
                except Exception as e:
                    print(f"Failed to open RealSense camera {dev}: {e}")
            else:
                try:
                    device_index = int(dev)
                    print(f"Ready to read deive: {device_index}")
                    cap = cv2.VideoCapture(device_index)

                    if self.cv_format == "MJPEG":
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
                    elif self.cv_format == "YUYV":
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))  # type: ignore[attr-defined]

                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)

                    if not cap.isOpened():
                        raise ValueError(f"Cannot open OpenCV camera {device_index}")

                    self.cameras.append({"type": "cv", "handle": cap})
                    print(f"OpenCV camera {device_index} opened successfully")
                except Exception as e:
                    print(f"Failed to open OpenCV camera {dev}: {e}")

    def get_images(self):
        images = []
        if len(self.cameras) == 0:
            # Return dummy images if no cameras available - use 640x480 which is expected by the model
            for _ in range(max(1, len(self.device_ids))):
                dummy_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                dummy_img[:, :, :] = 128  # Gray color instead of black
                images.append(dummy_img)
            return images

        for cam in self.cameras:
            if cam["type"] == "rs":
                try:
                    pipeline = cam["handle"]
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        dummy_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        dummy_img[:, :, :] = 128
                        images.append(dummy_img)
                    else:
                        img = np.asanyarray(color_frame.get_data())
                        images.append(img)
                except Exception as e:
                    print(f"Error reading from RealSense: {e}")
                    dummy_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    dummy_img[:, :, :] = 128
                    images.append(dummy_img)
            elif cam["type"] == "cv":
                cap = cam["handle"]
                ret, frame = cap.read()
                if not ret or frame is None:
                    dummy_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    dummy_img[:, :, :] = 128
                    images.append(dummy_img)
                else:
                    images.append(frame)
        return images

    def release(self):
        for cam in self.cameras:
            if cam["type"] == "rs":
                try:
                    cam["handle"].stop()
                except Exception:
                    pass
            elif cam["type"] == "cv":
                try:
                    cam["handle"].release()
                except Exception:
                    pass
        self.cameras = []


def normalization(state):
    """Normalize robot state for model input.

    - Threshold gripper: > 0.04 -> 1.0 else -1.0
    - Keep 6 joint values unchanged
    Returns np.ndarray of shape (7,)
    """
    arr = np.array(state, dtype=np.float32).copy()
    arr[6] = 1.0 if arr[6] > 0.04 else -1.0
    return arr


def unnormalization(action):
    """Unnormalize model action to robot command space.

    - Map gripper: > 0.5 -> 0.06850814 (open), else -> 0.025 (close)
    - Keep 6 joint values unchanged
    Returns np.ndarray of shape (7,)
    """
    arr = np.array(action, dtype=np.float32).copy()
    # arr[6] = 0.06850814 if arr[6] > 0.04 else 0.025
    arr[6] = 0.06850814 if arr[6] > 0.04 else 0.025
    return arr


class RobotWrapper:
    # Ports dict example: {'left_arm': 50051, 'right_arm': None}
    # Only arms with a non-None port will be initialized
    def __init__(self, url="localhost", ports=None, arm_speed="slow", type="move"):
        if ports is None:
            ports = {"left_arm": None, "right_arm": None}
        assert any(p is not None for p in ports.values()), "at least one arm port is required"
        assert arm_speed in [
            "slow",
            "default",
            "fast",
        ], "arm_speed must be in ['slow', default', 'fast']"
        assert type in ["move", "servo"], "type must be in ['move', 'servo']"
        self.type = type

        self.robots = {}

        for arm_name in ["left_arm", "right_arm"]:
            port = ports.get(arm_name)
            if port is None:
                continue

            robot = AIRBOTPlay(url=url, port=port)
            robot.connect()
            robot.set_speed_profile(SpeedProfile.SLOW if arm_speed == "slow" else SpeedProfile.FAST)

            product_info = robot.get_product_info()
            print("---------------------------------------------------------")
            print(f"Arm: {arm_name}")
            print(f"Product name: {product_info['product_type']}")
            print(f"Serial number: {product_info['sn']}")
            print(f"Simulation mode: {product_info['is_sim']}")
            print(f"Using interfaces: {product_info['interfaces']}")
            print(f"Installed end effectors: {product_info['eef_types']}")
            print(f"Firmware versions: {product_info['fw_versions']}")
            print("---------------------------------------------------------")

            # Default initial joints per arm (optional; can be customized)
            if arm_name == "left_arm":
                joints = [0.0, 0.0, 0.15, -1.7, 0.1, 1.7]
            else:
                joints = [0.0, 0.0, 0.15, 1.7, -0.1, -1.7]

            # using move mode to move to initial pose
            robot.switch_mode(RobotMode.PLANNING_POS)
            # Overwrite with a safe neutral pose
            joints = [
                0.00019073777366429567,
                0.17948424816131592,
                0.027656977996230125,
                1.4654383659362793,
                -0.3435187339782715,
                -1.4288166761398315,
            ]
            # joints = [0.03604944050312042, 0.17948424816131592, 0.029564354568719864, 1.6039139032363892, -0.3419928252696991, -1.5939955711364746]
            robot.move_to_joint_pos(joints)
            robot.move_eef_pos([0.06850814])
            print(f"arm: {arm_name}, joints: {joints}")

            if self.type == "move":
                robot.switch_mode(RobotMode.PLANNING_POS)
            elif self.type == "servo":
                robot.switch_mode(RobotMode.SERVO_JOINT_POS)

            init_joint_pos = robot.get_joint_pos()
            init_eef_pos = robot.get_eef_pos()
            print(f"[{arm_name}] init_joint_pos: {init_joint_pos}, init_eef_pos: {init_eef_pos}")
            self.robots[arm_name] = robot
            print(f"robot arm {arm_name} (port: {port}) init success!")
            time.sleep(2)

    def move_to_pos(self, pos, arm="right_arm"):
        assert arm in self.robots, f"arm '{arm}' not initialized"
        if self.type == "move":
            self.robots[arm].move_to_joint_pos(pos[:6], blocking=True)
            self.robots[arm].move_eef_pos([pos[6]], blocking=True)
        elif self.type == "servo":
            self.robots[arm].servo_joint_pos(pos[:6])
            self.robots[arm].servo_eef_pos([pos[6]])

    def get_joint_pos(self, arm="right_arm"):
        assert arm in self.robots, f"arm '{arm}' not initialized"
        return self.robots[arm].get_joint_pos()

    def get_eef_pos(self, arm="right_arm"):
        assert arm in self.robots, f"arm '{arm}' not initialized"
        return self.robots[arm].get_eef_pos()

    def get_state_pos(self, arm="right_arm"):
        assert arm in self.robots, f"arm '{arm}' not initialized"
        pos = self.robots[arm].get_joint_pos()
        eef_pos = self.robots[arm].get_eef_pos()
        result = pos + eef_pos
        return result


class ActionSmoother:
    def __init__(self, method="exponential", alpha=0.1, window_size=7, smooth_dims=None):
        self.method = method
        self.window_size = window_size
        self.smooth_dims = smooth_dims

        if isinstance(alpha, (list, tuple, np.ndarray)):
            self.alpha = np.array(alpha, dtype=np.float32)
        else:
            self.alpha = alpha

        self.history = []
        self.smoothed_action = None

    def smooth_action(self, raw_action):
        raw_action = np.array(raw_action, dtype=np.float32)

        self.history.append(raw_action.copy())
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size :]

        if self.smoothed_action is None:
            self.smoothed_action = raw_action.copy()
            return self.smoothed_action

        result_action = raw_action.copy()

        if self.smooth_dims is None:
            dims_to_smooth = list(range(len(raw_action)))
        else:
            dims_to_smooth = [d for d in self.smooth_dims if d < len(raw_action)]

        if self.method == "exponential":
            if isinstance(self.alpha, np.ndarray):
                for dim in dims_to_smooth:
                    alpha = self.alpha[dim] if dim < len(self.alpha) else self.alpha[-1]
                    result_action[dim] = (
                        alpha * raw_action[dim] + (1 - alpha) * self.smoothed_action[dim]
                    )
            else:
                for dim in dims_to_smooth:
                    result_action[dim] = (
                        self.alpha * raw_action[dim] + (1 - self.alpha) * self.smoothed_action[dim]
                    )

        elif self.method == "moving_average":
            history_array = np.array(self.history)
            for dim in dims_to_smooth:
                result_action[dim] = np.mean(history_array[:, dim])

        elif self.method == "linear_interpolation":
            for dim in dims_to_smooth:
                result_action[dim] = 0.7 * raw_action[dim] + 0.3 * self.smoothed_action[dim]

        elif self.method == "identity":
            return result_action

        elif self.method == "average":
            history_array = np.array(self.history)
            for dim in dims_to_smooth:
                result_action[dim] = np.mean(history_array[:, dim])

        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

        self.smoothed_action = result_action.copy()

        return result_action

    def reset(self):
        self.history = []
        self.smoothed_action = None


class VLAClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def predict(self, obs):
        try:
            response = requests.post(
                self.server_url,
                json={"observation": obs},
            )
            action_chunk = response.json()

            actions = []
            for arm, gripper in zip(
                action_chunk["action.right_arm"], action_chunk["action.right_gripper"]
            ):
                action = np.asarray(list(arm) + [float(gripper)], dtype=np.float32)
                actions.append(action)

            return np.array(actions)
        except Exception as e:
            print(f"VLA prediction error: {e}")
            return None


def predict_actions(server_url, obs):
    response = requests.post(
        server_url,
        json={"observation": obs},
    )
    return response.json()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_rtc", action="store_true", default=False)
    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument(
        "--instruction",
        default="Pick up each block one by one and place them all into the bowl.",
        type=str,
    )
    args = parser.parse_args()

    USE_RTC_CONTROLLER = args.use_rtc
    print(f"Use RTC Controller: {USE_RTC_CONTROLLER}")
    print(f"Instruction: {args.instruction}")

    task_description = "Pick up each block one by one and place them all into the bowl."
    task_description = "Pick up each block one by one and place them all into the blue bowl."
    task_description = "Pick up each block one by one and place them all into the right bowl."
    task_description = "Pick up 3 purple blocks, stack them in a 3-layer tower, and place it at the center of the red line."
    # task_description = 'Pick up three blue blocks and stack them on the red point to form a three-layer tower.'
    task_description = (
        "Pick up two blue blocks and stack them on the red point to form a two-layer tower."
    )
    task_description = (
        # "Pick up 3 blue blocks and stack them on the red point to form a 3 layer tower."
        # "Pick up 3 blocks and stack them on the red point to form a 3-layer tower."
        "ick up 3 blue block and place it on the red point to form a 3-block tower."
    )

    # task_description = 'Pick up one blue block and place it on the red point to form a one-block tower.'
    # task_description = 'Pick up two blue blocks and stack them on the red point to form a two-layer tower.'
    # task_description = "Pick up three blocks: gray for the first layer, orange for the second, and purple for the third, then stack them at the center of the red line."
    # task_description = args.instruction

    print("Robot System Init...")
    robots = RobotWrapper(
        url="localhost",
        ports={"left_arm": None, "right_arm": 50053},
        arm_speed="default" if args.fast else "slow",
        type="servo",
    )

    if args.fast:
        logger.info("Using fast (default) servo mode for right arm")
        target_queue = queue.Queue(maxsize=30)
        servo_queue = queue.Queue(maxsize=30)
        rp = RuckigPlanner(
            robots.robots["right_arm"], robots.get_state_pos("right_arm"), DoFs=7, dt=0.02
        )
        rp.start(servo_queue, target_queue)

    # server_url = "http://106.13.248.32:10090/act"
    server_url = "http://127.0.0.1:10090/act"
    # server_url = "http://114.111.24.161:10090/act"
    # server_url = "http://106.13.245.8:10091/act"
    print(f"VLA Server URL: {server_url}")

    # Camera System Init
    print("Camera System Init...")

    caps = CameraWrapper(
        devices=["215322074711", "242622070332", 6],
        num_realsense=2,
        # width=640,
        # height=480,
        width=1280,
        height=720,
        fps=30,
        cv_format="MJPEG",
    )

    time.sleep(2)

    print(f"Smoother System Init for Grippers...")
    action_smoother = ActionSmoother(method="average", alpha=0.1, window_size=10, smooth_dims=[6])
    # action_smoother = None

    print("Loop Start!")
    step_count = 0

    from collections import deque

    # we use only 12 from 16
    chunk_size = 14
    cached_actions = deque(maxlen=chunk_size)

    while True:
        try:
            loop_start_time = time.time()
            step_count += 1

            # 如果所有actions都执行完了，获取新的chunk
            # if current_action_idx >= len(current_actions):
            if len(cached_actions) == 0:
                print(f"[Sync Mode] Obtain new action chunk...")

                # image
                images = caps.get_images()
                images = caps.get_images()
                images = caps.get_images()

                if len(images) >= 3:
                    img_right, img_front, img_env = images[:3]
                    img_obs = np.concatenate([img_right, img_front, img_env], axis=1)
                    cv2.imwrite(f"outputs/img_obs.jpg", img_obs)
                    print(f"image.shape: {img_right.shape}, {img_front.shape}, {img_env.shape}")
                else:
                    filler = np.zeros((1280, 720, 3), dtype=np.uint8)
                    filler[:, :, :] = 128
                    imgs = images + [filler] * (3 - len(images))
                    img_right, img_front, img_env = imgs[:3]

                # state
                # arm_state = normalization(robots.get_state_pos(arm="right_arm"))
                arm_state = robots.get_state_pos(arm="right_arm")

                # obs
                obs = {
                    "video.cam_head": img_front[np.newaxis, ::],
                    "video.cam_env": img_env[np.newaxis, ::],
                    "video.cam_right_wrist": img_right[np.newaxis, ::],
                    "state.right_arm": np.expand_dims(
                        np.array(arm_state[:6], dtype=np.float32), axis=0
                    ),
                    "state.right_gripper": np.expand_dims(
                        np.array([arm_state[6]], dtype=np.float32), axis=0
                    ),
                    "annotation.human.task_description": [task_description],
                }

                start_time = time.time()
                action_chunk = predict_actions(server_url, obs)
                logger.info(f"inference time cost: {time.time() - start_time:.3f}s")

                # current_actions = []
                # current_action_idx = 0
                for arm, gripper in zip(
                    action_chunk["action.right_arm"], action_chunk["action.right_gripper"]
                ):
                    action = np.asarray(list(arm) + [float(gripper)], dtype=np.float32)
                    # current_actions.append(action)
                    cached_actions.append(action)

                print(f"Obtainbed {len(cached_actions)} action steps...")

            # if current_action_idx < len(current_actions):
            #     raw_action = current_actions[current_action_idx]
            #     current_action_idx += 1

            if len(cached_actions) > 0:
                smoothed_action = action_smoother.smooth_action(cached_actions[0])
                denormalized_action = unnormalization(smoothed_action)
                robots.move_to_pos(denormalized_action.tolist(), arm="right_arm")
                cached_actions.popleft()
                # print(
                #     f"[Sync Mode] Step {step_count}: action_idx={current_action_idx}/{len(current_actions)}"
                # )
                # print(f"   Raw: {raw_action}")

            # 20Hz control
            time.sleep(0.13)

        except KeyboardInterrupt:
            print("Received interrupt signal, exiting safely...")
            break
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"SYNC control mode error: {e}, might have issues in server response!")
            time.sleep(0.1)

    print("Clearing resources...")
    try:
        caps.release()
        print("Camera resources released")
    except Exception as e:
        print(f"Camera clearing error: {e}")

    print("Program ended")
