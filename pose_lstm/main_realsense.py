# main_realsense.py
import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import time
import json
import threading
import pandas as pd
from datetime import datetime
from collections import deque, Counter
from tensorflow.keras.models import load_model  
from config import *

RECORD_MODE = False  

# 全域共享變數
current_state = "STANDING"
pose_start_time = None
camId = 1
last_sent_pose = None
display_running = True

# 基礎物理特徵平滑快取
SMOOTHING_WINDOW = 8
knee_angle_buf = deque(maxlen=SMOOTHING_WINDOW)
hip_depth_diff_buf = deque(maxlen=SMOOTHING_WINDOW)
knee_diff_buf = deque(maxlen=SMOOTHING_WINDOW)
foot_height_diff_buf = deque(maxlen=SMOOTHING_WINDOW)

# LSTM 快取
lstm_feature_window = deque(maxlen=N_TIME)
pose_history = deque(maxlen=20)
ai_predicted_label = "None"

classes = []
model = None
record_label = "Unknown"
recorded_data = []

# ==========================================================
# 🚀 初始化 MediaPipe 與 RealSense
# ==========================================================
print("⏳ 正在初始化 MediaPipe 骨架辨識模組...")
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("⏳ 正在啟動 RealSense 攝影機硬體管線...")
pipeline = rs.pipeline()
config_rs = rs.config()
config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config_rs)
align = rs.align(rs.stream.color)

PROCESS_EVERY = 2
last_results = None
frame_count = 0

# ==========================================================
# 🚀 錄製/預測模式選單與「文字端姿勢校正」
# ==========================================================
start_recording_signal = False

if RECORD_MODE:
    print("\n=== 🛠️ ITRI 房務動作資料錄製選單 ===")
    print("1) SL_forward_stepping (跨步)")
    print("2) knee_propping (鋪床)")
    print("3) squat (蹲下)")
    print("4) STANDING (標準站立)")
    print("====================================")
    
    while True:
        choice = input("✏️ 請選擇這次要錄製的動作編號 (1-4): ").strip()
        if choice == '1': record_label = "SL_forward_stepping"; break
        elif choice == '2': record_label = "knee_propping"; break
        elif choice == '3': record_label = "squat"; break
        elif choice == '4': record_label = "STANDING"; break
        else: print("❌ 輸入錯誤！請輸入 1, 2, 3 或 4")

    print(f"\n🎯 確認錄製標籤: 【{record_label}】")
    print("📢 [盲錄文字校正模式]：請走到相機前方，稍等一下，下方會即時顯示相機有沒有抓到你的身體骨架...")
    
    # 進入一個文字校正迴圈，直到使用者滿意按下 Enter
    check_lock = threading.Event()
    
    def alignment_check_thread():
        global frame_count, last_results
        print("\n=== 🚶‍♂️ 骨架即時偵測測試（請站在相機前） ===")
        while not check_lock.is_set():
            try:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                if not color_frame: continue
                
                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = pose_model.process(rgb_image)
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    required = [
                        mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                        mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value,
                    ]
                    if min(lm[i].visibility for i in required) >= 0.4:
                        print(f"\r🟢 [OK] 骨架鎖定成功！ 左右膝蓋可見度佳。   ", end="", flush=True)
                    else:
                        print(f"\r🟡 [WARN] 雖有拍到人，但下半身/膝蓋被擋住或看不清！  ", end="", flush=True)
                else:
                    print(f"\r🔴 [ERROR] 畫面上沒有任何人影！請站到鏡頭前。  ", end="", flush=True)
                time.sleep(0.1)
            except:
                break
                
        print("\n=== 校正結束 ===")

    t_check = threading.Thread(target=alignment_check_thread, daemon=True)
    t_check.start()
    
    input("\n👉 當你看到上方顯示 🟢 [OK] 且站好預備姿勢後，請在此終端機按下 [Enter] 鍵開始倒數錄製...")
    check_lock.set() # 停止校正文字
    t_check.join(timeout=1)
    
    print(f"⏳ 倒數 {N_DELAY} 秒後正式記錄數據...")
    for i in range(N_DELAY, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("🚀 開始錄製！請重覆或維持該房務動作...")
    start_recording_signal = True
else:
    model_path = os.path.join(MODEL_DIR, 'best.h5')
    if os.path.exists(model_path):
        model = load_model(model_path)
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        classes = sorted([f.split('.')[0] for f in files])
        print(f"🤖 Keras LSTM 模型載入成功！行為類別: {classes}")

# ==========================================================
# 異步 LSTM 推理執行執行緒
# ==========================================================
def async_lstm_inference(feature_snapshot):
    global ai_predicted_label
    if model is None: return
    tensor = np.expand_dims(feature_snapshot, axis=0)
    result = model.predict(tensor, verbose=0)
    ai_predicted_label = classes[np.argmax(result[0])]

def get_region_depth(depth_frame, cx, cy, radius=3):
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if cx < 0 or cx >= dw or cy < 0 or cy >= dh: return 0.0
    depths = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < dw and 0 <= ny < dh:
                d = depth_frame.get_distance(nx, ny)
                if d > 0.01: depths.append(d)
    return float(np.median(depths)) if len(depths) >= 3 else 0.0

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ==========================================================
# 主循環影像處理管線 (純文字背景運作)
# ==========================================================
try:
    while display_running:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame: continue

        frame_count += 1
        color_image = np.asanyarray(color_frame.get_data())
        h, w, _ = color_image.shape
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY == 0:
            results = pose_model.process(rgb_image)
            last_results = results
        else:
            results = last_results

        detected_pose = "STANDING"
        knee_angle = hip_depth_diff = knee_diff = duration = 0.0

        if results is not None and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            required = [
                mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value,
            ]

            if min(lm[i].visibility for i in required) >= 0.4:
                def lm_px(idx): return int(lm[idx].x * w), int(lm[idx].y * h)

                r_hip, l_hip = lm_px(mp_pose.PoseLandmark.RIGHT_HIP.value), lm_px(mp_pose.PoseLandmark.LEFT_HIP.value)
                r_knee, l_knee = lm_px(mp_pose.PoseLandmark.RIGHT_KNEE.value), lm_px(mp_pose.PoseLandmark.LEFT_KNEE.value)
                r_ank, l_ank = lm_px(mp_pose.PoseLandmark.RIGHT_ANKLE.value), lm_px(mp_pose.PoseLandmark.LEFT_ANKLE.value)

                foot_height_diff_raw = abs(r_ank[1] - l_ank[1])
                r_hip_d = get_region_depth(depth_frame, *r_hip)
                l_hip_d = get_region_depth(depth_frame, *l_hip)

                if r_hip_d > 0.0 and l_hip_d > 0.0:
                    r_ka = calculate_angle(r_hip, r_knee, r_ank)
                    l_ka = calculate_angle(l_hip, l_knee, l_ank)
                    
                    knee_angle_raw = (r_ka + l_ka) / 2
                    hip_depth_diff_raw = abs(r_hip_d - l_hip_d)
                    knee_diff_raw = abs(r_ka - l_ka)

                    knee_angle_buf.append(knee_angle_raw)
                    hip_depth_diff_buf.append(hip_depth_diff_raw)
                    knee_diff_buf.append(knee_diff_raw)
                    foot_height_diff_buf.append(foot_height_diff_raw)

                    knee_angle = np.mean(knee_angle_buf)
                    hip_depth_diff = np.mean(hip_depth_diff_buf)
                    knee_diff = np.mean(knee_diff_buf)
                    foot_height_diff = np.mean(foot_height_diff_buf)
                    knee_angle_std = float(np.std(list(knee_angle_buf))) if len(knee_angle_buf) >= 4 else 0.0

                    current_features = [
                        knee_angle, hip_depth_diff, knee_diff, foot_height_diff, knee_angle_std,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                        lm[mp_pose.PoseLandmark.NOSE.value].y
                    ]

                    if RECORD_MODE:
                        if start_recording_signal:
                            recorded_data.append(current_features)
                            # 每 50 幀在終端機報一次進度
                            if len(recorded_data) % 50 == 0:
                                print(f"📊 已錄製特徵幀數: {len(recorded_data)} / {N_FRAME}")
                                
                            if len(recorded_data) >= N_FRAME:
                                df = pd.DataFrame(recorded_data)
                                df.to_csv(os.path.join(DATA_DIR, f"{record_label}.csv"), index=False)
                                print(f"\n📁 【錄製完成】！數據已成功儲存至: data/{record_label}.csv")
                                display_running = False
                    else:
                        # 預測模式邏輯 (維持不變)
                        lstm_feature_window.append(current_features)
                        if len(lstm_feature_window) == N_TIME:
                            feature_snapshot = np.array(list(lstm_feature_window))
                            t = threading.Thread(target=async_lstm_inference, args=(feature_snapshot,))
                            t.start()
                        pose_history.append(ai_predicted_label if ai_predicted_label != "None" else "STANDING")
                        detected_pose = Counter(pose_history).most_common(1)[0][0]
                        if current_state != detected_pose:
                            current_state = detected_pose
                            print(f"\n📡 [AI 預測狀態變更] → {current_state} (膝蓋角度: {knee_angle:.1f}°)")

except KeyboardInterrupt:
    print("\n👋 程式已由使用者手動終止。")
finally:
    pipeline.stop()