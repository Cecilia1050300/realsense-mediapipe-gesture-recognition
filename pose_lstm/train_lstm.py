# train_lstm.py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from config import *

def load_data():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    classes = sorted([f.split('.')[0] for f in files])
    
    df_dict = {}
    for className in classes:
        df_dict[className] = pd.read_csv(os.path.join(DATA_DIR, f"{className}.csv"))
    return classes, df_dict

def encode_data(classes, df_dict):
    nClass = len(classes)
    X, y = [], []

    for idx, className in enumerate(classes):
        df = df_dict[className]
        nSample = len(df)
        # 核心亮點：用滑動視窗切割連續的 N_TIME 幀動作軌跡趨勢
        for start in range(nSample - N_TIME):
            X.append(df.iloc[start : start + N_TIME, :].values)
            one_hot = [0] * nClass
            one_hot[idx] = 1
            y.append(one_hot)

    return np.array(X), np.array(y)

def build_model(input_shape, num_classes):
    # 標準 4 層疊加 LSTM (Stacked LSTM) 網路架構
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("📦 正在載入動作資料集...")
    classes, df_dict = load_data()
    if not classes:
        print("❌ 找不到任何訓練資料！請先將 main_realsense.py 的 RECORD_MODE 設為 True 進行錄製。")
        return
        
    print(f"🎯 偵測到行為類別: {classes}")
    X, y = encode_data(classes, df_dict)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print("🏗️ 正在初始化 LSTM 神經網路...")
    model = build_model(input_shape, len(classes))
    
    print("🚀 開始訓練 AI 行為模型...")
    history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    
    model_path = os.path.join(MODEL_DIR, 'best.h5')
    model.save(model_path)
    print(f"🎉 訓練完成！最佳權重大腦已儲存至: {model_path}")

if __name__ == "__main__":
    main()
    