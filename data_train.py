import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from data_loader import load_timeseries
import datetime

# === 설정 ===
MODE = "monthly"  # or "merged"
MODEL_TYPE = "hybrid"  # "hybrid" or "timeseries_only"
EPOCHS = 30
BATCH_SIZE = 32

# === 데이터 로드 ===
X, meta, y, label_encoder = load_timeseries(mode=MODE)

if MODEL_TYPE == "timeseries_only":
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    X_train, X_val, meta_train, meta_val, y_train, y_val = train_test_split(
        X, meta, y, test_size=0.2, random_state=42, stratify=y
    )

# === 모델 정의 ===
def build_hybrid_model(input_shape_ts, input_shape_meta, num_classes):
    ts_input = Input(shape=input_shape_ts, name="timeseries_input")
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

    meta_input = Input(shape=input_shape_meta, name="meta_input")
    m = Dense(16, activation="relu")(meta_input)

    combined = Concatenate()([x, m])
    dense = Dense(64, activation="relu")(combined)
    dense = Dropout(0.4)(dense)
    output = Dense(num_classes, activation="softmax")(dense)

    return Model(inputs=[ts_input, meta_input], outputs=output)

def build_timeseries_only_model(input_shape_ts, num_classes):
    ts_input = Input(shape=input_shape_ts, name="timeseries_input")
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
    dense = Dense(64, activation="relu")(x)
    dense = Dropout(0.4)(dense)
    output = Dense(num_classes, activation="softmax")(dense)

    return Model(inputs=ts_input, outputs=output)

# === 모델 빌드 ===
if MODEL_TYPE == "hybrid":
    model = build_hybrid_model((X.shape[1], 1), (meta.shape[1],), y.shape[1])
else:
    model = build_timeseries_only_model((X.shape[1], 1), y.shape[1])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# === 학습 ===
if MODEL_TYPE == "hybrid":
    history = model.fit(
        [X_train, meta_train], y_train,
        validation_data=([X_val, meta_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
else:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

# === 결과 평가 ===
if MODEL_TYPE == "hybrid":
    y_pred = model.predict([X_val, meta_val])
else:
    y_pred = model.predict(X_val)

y_true_labels = np.argmax(y_val, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_true_labels, y_pred_labels)
macro_f1 = f1_score(y_true_labels, y_pred_labels, average="macro")
report_str = classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_)

# === 학습 곡선 저장 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title(f'Accuracy ({MODEL_TYPE})')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'Loss ({MODEL_TYPE})')
plt.legend()
plt.tight_layout()
plt.savefig(f"training_curve_{MODEL_TYPE}.png")
plt.close()

# === 모델 저장 ===
model_name = f"saved_model_{MODEL_TYPE}.h5"
model.save(model_name)

# === 결과 레포트 저장 ===
now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(f"report_{MODEL_TYPE}.txt", "w", encoding="utf-8") as f:
    f.write(f"모델 타입: {MODEL_TYPE}\n")
    f.write(f"데이터 모드: {MODE}\n")
    f.write(f"학습일시: {now_str}\n")
    f.write(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}\n")
    f.write(f"Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    f.write(f"Train Loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"Overall Accuracy: {acc:.4f}\n")
    f.write(f"Macro F1 Score: {macro_f1:.4f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(report_str)
    f.write("\n\n모델 저장 경로: " + model_name + "\n")
    f.write("레이블 목록: " + ", ".join(label_encoder.classes_) + "\n")

print(f"Model saved: {model_name}")
print(f"Report saved: report_{MODEL_TYPE}.txt")
print(f"Training curve saved: training_curve_{MODEL_TYPE}.png")
