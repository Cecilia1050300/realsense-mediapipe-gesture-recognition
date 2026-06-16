# config.py
import os

# 時序參數：AI 每次看過去幾幀的連續動作（30 幀大約是 1 秒鐘的連續動態）
N_TIME = 30  

# 錄製模式下，收集每筆資料時的幀數上限（1500 幀大約可以錄製 50 秒的訓練集）
N_FRAME = 1500  

# 開始錄製前的倒數準備秒數
N_DELAY = 3  

# AI 訓練超參數
N_EPOCH = 30
BATCH_SIZE = 32

# 目錄路徑定義
DATA_DIR = "./data"
MODEL_DIR = "./models"

# 自動確保資料夾存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)