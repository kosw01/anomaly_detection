import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_loader import load_timeseries

# === 데이터 로드 ===
MODE = "monthly"  # or "merged"
X, meta, y, label_encoder = load_timeseries(mode=MODE)

# 학습/검증 분할
X_train, X_val, meta_train, meta_val, y_train, y_val = train_test_split(
    X, meta, y, test_size=0.2, random_state=42, stratify=y
)

# === 모델 정의 ===
# 시계열 입력
ts_input = Input(shape=(X.shape[1], 1), name="timeseries_input")
x = Conv1D(32, kernel_size=5, padding="same")(ts_input)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv1D(64, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Flatten()(x)

# 메타데이터 입력
meta_input = Input(shape=(meta.shape[1],), name="meta_input")
m = Dense(16, activation="relu")(meta_input)

# 결합
combined = Concatenate()([x, m])
dense = Dense(64, activation="relu")(combined)
dense = Dropout(0.4)(dense)
output = Dense(y.shape[1], activation="softmax")(dense)

model = Model(inputs=[ts_input, meta_input], outputs=output)

# === 학습 설정 ===
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# === 학습 ===
history = model.fit(
    [X_train, meta_train], y_train,
    validation_data=([X_val, meta_val], y_val),
    epochs=30,
    batch_size=32
)

# === 학습 결과 시각화 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# === 모델 저장 ===
model.save("saved_model.h5")
print(f"Model saved. Classes: {label_encoder.classes_}")
