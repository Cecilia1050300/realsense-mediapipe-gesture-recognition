import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import time
import json
import os
import threading
from datetime import datetime
from collections import deque, Counter

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

# ==========================================================
# 全域共享變數
# ==========================================================
latest_frame = None
frame_lock = threading.Lock()
display_running = True

current_state = "STANDING"
pose_start_time = None
camId = 1
last_sent_pose = None
last_printed_state = "STANDING"

SMOOTHING_WINDOW = 8
knee_angle_buf = deque(maxlen=SMOOTHING_WINDOW)
hip_depth_diff_buf = deque(maxlen=SMOOTHING_WINDOW)

knee_diff_buf = deque(maxlen=SMOOTHING_WINDOW)
foot_height_diff_buf = deque(maxlen=SMOOTHING_WINDOW)

pose_history = deque(maxlen=20)


# ==========================================================
# 顯示 Thread
# ==========================================================
def display_thread_func():
    global display_running
    cv2.namedWindow('ITRI Pose Tracking', cv2.WINDOW_GUI_NORMAL)

    # 等待畫面還沒來時的佔位畫面
    waiting = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(waiting, "Waiting for camera...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    while display_running:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else waiting

        cv2.imshow('ITRI Pose Tracking', frame)

        key = cv2.waitKey(66) & 0xFF
        if key == ord('q'):
            display_running = False
            break

        try:
            if cv2.getWindowProperty('ITRI Pose Tracking', cv2.WND_PROP_VISIBLE) < 1:
                display_running = False
                break
        except cv2.error:
            display_running = False
            break

    cv2.destroyAllWindows()


# ==========================================================
# 工具函式
# ==========================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def get_region_depth(depth_frame, cx, cy, radius=3):
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if cx < 0 or cx >= dw or cy < 0 or cy >= dh:
        return 0.0
    depths = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < dw and 0 <= ny < dh:
                d = depth_frame.get_distance(nx, ny)
                if d > 0.01:
                    depths.append(d)
    return float(np.median(depths)) if len(depths) >= 3 else 0.0



def classify_pose(knee_angle, hip_depth_diff, knee_diff, knee_angle_std=0.0):

    # ① SL_forward_stepping（跨步）
    # 膝角較淺 135~168°，膝差 >= 12°
    if 135 <= knee_angle <= 168 and knee_diff >= 12:
        return "SL_forward_stepping"

    # ② knee_propping（鋪床）
    # 膝角較深 118~135°，膝差放寬到 >= 8°
    if 118 <= knee_angle <= 135 and knee_diff >= 8:
        return "knee_propping"

    # ③ squat（蹲下）
    if knee_angle < 118 and knee_diff < 15:
        return "squat"

    return "STANDING"


def draw_overlay(image, state, duration, knee_angle, hip_diff, knee_diff, detected_pose):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (640, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    color_map = {
        "STANDING":            (200, 200, 200),
        "squat":               (0, 255, 100),
        "SL_forward_stepping": (0, 180, 255),
        "knee_propping":       (255, 140, 0),
    }
    color = color_map.get(state, (255, 255, 255))

    cv2.putText(image, f"STATE: {state} ({duration:.1f}s)",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # 這裡改成印出：Knee(平均膝蓋角度)、KneeDiff(左右膝差)、HipDiff(髖深度差)
    cv2.putText(image, f"Knee:{knee_angle:.1f}  KneeDiff:{knee_diff:.1f}  HipDiff:{hip_diff:.3f}m",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(image, f"Detected: {detected_pose}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)
    return image


# ==========================================================
# 主程式
# ==========================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

PROCESS_EVERY = 2
last_results = None

display_thread = threading.Thread(target=display_thread_func, daemon=True)
display_thread.start()
print("🚀 姿態辨識啟動 | Ctrl+C 或按 Q 結束\n")

frame_count = 0

try:
    while display_running:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame_count += 1
        color_image = np.asanyarray(color_frame.get_data())
        h, w, _ = color_image.shape

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY == 0:
            results = pose_model.process(rgb_image)
            last_results = results
        else:
            results = last_results

        # 每幀預設值（當完全沒偵測到人，或深度遺失時，UI 顯示會自動歸零，防止數值卡死）
        detected_pose = "STANDING"
        knee_angle = hip_depth_diff = knee_diff = duration = 0.0

        if results is not None and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            # ---------------------------
            # Visibility 檢查
            # ---------------------------
            required = [
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value,
            ]

            if min(lm[i].visibility for i in required) < 0.4:
                pose_history.append(current_state)  # 保持狀態
                detected_pose = current_state
            else:
                def lm_px(idx):
                    return int(lm[idx].x * w), int(lm[idx].y * h)

                r_sho  = lm_px(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                l_sho  = lm_px(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                r_hip  = lm_px(mp_pose.PoseLandmark.RIGHT_HIP.value)
                l_hip  = lm_px(mp_pose.PoseLandmark.LEFT_HIP.value)
                r_knee = lm_px(mp_pose.PoseLandmark.RIGHT_KNEE.value)
                l_knee = lm_px(mp_pose.PoseLandmark.LEFT_KNEE.value)
                r_ank  = lm_px(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
                l_ank  = lm_px(mp_pose.PoseLandmark.LEFT_ANKLE.value)

                foot_height_diff_raw = abs(r_ank[1] - l_ank[1])

                r_hip_d = get_region_depth(depth_frame, *r_hip)
                l_hip_d = get_region_depth(depth_frame, *l_hip)

                # 髖有深度才運算
                if r_hip_d > 0.0 and l_hip_d > 0.0:
                    r_ka = calculate_angle(r_hip, r_knee, r_ank)
                    l_ka = calculate_angle(l_hip, l_knee, l_ank)
                    knee_angle_raw     = (r_ka + l_ka) / 2
                    hip_depth_diff_raw = abs(r_hip_d - l_hip_d)
                    knee_diff_raw = abs(r_ka - l_ka)

                    knee_angle_buf.append(knee_angle_raw)
                    hip_depth_diff_buf.append(hip_depth_diff_raw)
                    knee_diff_buf.append(knee_diff_raw)
                    foot_height_diff_buf.append(foot_height_diff_raw)

                    # 計算平滑快取平均值
                    knee_angle = np.mean(knee_angle_buf)
                    hip_depth_diff = np.mean(hip_depth_diff_buf)
                    knee_diff = np.mean(knee_diff_buf)
                    foot_height_diff = np.mean(foot_height_diff_buf)
                    
                    # 計算膝蓋角度標準差（動態波動度）
                    knee_angle_std = float(np.std(list(knee_angle_buf))) if len(knee_angle_buf) >= 4 else 0.0

                    # 進行決策
                    detected_pose = classify_pose(
                        knee_angle=knee_angle,
                        hip_depth_diff=hip_depth_diff,
                        knee_diff=knee_diff,
                        knee_angle_std=knee_angle_std
                    )

                    if frame_count % 10 == 0:
                        print(f"[#{frame_count:05d}] 膝:{knee_angle:5.1f}°(波動:{knee_angle_std:.1f}) "
                              f"膝差:{knee_diff:.1f}° 髖差:{hip_depth_diff:.3f}m → {detected_pose}")

                    if detected_pose != last_printed_state:
                        print(f"\n📡 切換: {last_printed_state} → {detected_pose} | "
                              f"膝:{knee_angle:.1f}°(波動:{knee_angle_std:.1f}) 膝差:{knee_diff:.1f}° 髖差:{hip_depth_diff:.3f}m\n")
                        last_printed_state = detected_pose

                    pose_history.append(detected_pose)

                    major_pose = Counter(pose_history).most_common(1)[0][0]

                    if current_state != major_pose:
                        current_state = major_pose
                        pose_start_time = time.time()

                    if pose_start_time is not None:
                        duration = time.time() - pose_start_time
                        if (duration >= 1.2
                                and last_sent_pose != current_state
                                and current_state != "STANDING"):
                            action_json = {
                                "camId": camId,
                                "success_pose": current_state,
                                "timestamp": datetime.now().isoformat()
                            }
                            print(f"\n🔥 【發送 MQTT】: {json.dumps(action_json)}\n")
                            last_sent_pose = current_state

                    if current_state == "STANDING":
                        last_sent_pose = None
                else:
                    # 有骨架但剛好深度的光點沒打到屁股上時，維持當前穩定狀態
                    detected_pose = current_state

        # 計算當前狀態的持續時間 (用於畫面上方顯示)
        if pose_start_time is not None and current_state != "STANDING":
            duration = time.time() - pose_start_time
        else:
            duration = 0.0

        # ✅ 不管有沒有偵測到人，每幀都推給顯示 Thread（修正變數參數，引入 knee_diff）
        color_image = draw_overlay(color_image, current_state, duration,
                                   knee_angle, hip_depth_diff, knee_diff, detected_pose)
        with frame_lock:
            latest_frame = color_image

except KeyboardInterrupt:
    print("\n👋 程式已終止。")
finally:
    display_running = False
    display_thread.join(timeout=2)
    pipeline.stop()