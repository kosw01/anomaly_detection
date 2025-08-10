import pandas as pd
import numpy as np
import os

MAX_SEQ_LEN = 4300

def load_timeseries(mode="monthly"):
    info_df = pd.read_csv("channel_info/channel_info.csv")
    bridges = info_df['br_name'].unique()
    
    X, meta, y = [], [], []
    
    for _, row in info_df.iterrows():
        br_name = row['br_name']
        ch_name = row['channel_name']
        
        months = ["2025-05", "2025-06", "2025-07"]
        monthly_data = []
        monthly_labels = []
        
        for month in months:
            file_path = f"{month}{br_name}/{br_name}_통계데이터.csv"
            if not os.path.exists(file_path):
                continue
            
            df = pd.read_csv(file_path)
            col = f"{ch_name}_AVG"
            if col not in df.columns:
                continue
            
            ts = df[col].values
            
            # 길이 조정
            if len(ts) > MAX_SEQ_LEN:
                ts = ts[-MAX_SEQ_LEN:]  # 월 초부분 잘라냄
            elif len(ts) < MAX_SEQ_LEN:
                ts = np.pad(ts, (MAX_SEQ_LEN - len(ts), 0), 'constant')
            
            monthly_data.append(ts)
            
            # 라벨 선택
            label_col = f"result_{month.split('-')[1].capitalize()}"
            monthly_labels.append(row[label_col])
        
        # mode별 처리
        if mode == "monthly":
            for ts, lbl in zip(monthly_data, monthly_labels):
                X.append(ts)
                meta.append(row[['channel_type', 'supperposition']].values)
                y.append(lbl)
        elif mode == "merged":
            merged_ts = np.concatenate(monthly_data, axis=0)
            if len(merged_ts) > MAX_SEQ_LEN:
                merged_ts = merged_ts[-MAX_SEQ_LEN:]
            elif len(merged_ts) < MAX_SEQ_LEN:
                merged_ts = np.pad(merged_ts, (MAX_SEQ_LEN - len(merged_ts), 0), 'constant')
            
            # 3개월 중 가장 많이 나온 라벨 선택
            merged_label = max(set(monthly_labels), key=monthly_labels.count)
            X.append(merged_ts)
            meta.append(row[['channel_type', 'supperposition']].values)
            y.append(merged_label)
    
    X = np.array(X).reshape(len(X), MAX_SEQ_LEN, 1)
    meta = np.array(meta)
    
    # 라벨 인코딩
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_onehot = OneHotEncoder(sparse=False).fit_transform(y_enc.reshape(-1, 1))
    
    return X, meta, y_onehot, le
