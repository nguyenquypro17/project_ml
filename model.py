#PHẦN 1: CÀI ĐẶT MÔI TRƯỜNG (CHẠY ĐẦU TIÊN)
import os
import sys
os.system("pip install 'protobuf==3.20.3' 'rdkit>=2023.3.3' > /dev/null 2>&1")
print(">>> Đã cài đặt xong môi trường!")
print(tf.config.list_physical_devices('GPU'))
# --- PHẦN 2: IMPORT THƯ VIỆN & CẤU HÌNH ---
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, Dropout, BatchNormalization, MaxPooling1D, Dense, LSTM, 
                                     Bidirectional, Activation)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score)
import matplotlib.pyplot as plt

def load_data_from_npz(npz_path, num_classes=86):
    print(f"--- Đang tải dữ liệu từ: {npz_path} ---")
    data = np.load(npz_path)
    # 1. Load X (Đã là 4096 features từ preprocess)
    X = data['X']
    # Reshape cho Conv1D: (N, 4096) -> (N, 4096, 1)
    X = np.expand_dims(X, axis=-1)
    # 2. Load y
    y = data['y']
    # One-hot encoding
    y = to_categorical(y, num_classes=num_classes)
    print(f" > Shape X: {X.shape} | Shape y: {y.shape}")
    return X, y

def my_model(input_shape, num_classes):
    model = Sequential([
        # CNN Block 1
        Conv1D(filters=16, kernel_size=3, input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        Dropout(0.1),
        # CNN Block 2
        Conv1D(filters=32, activation='relu', kernel_size=3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),
        # CNN Block 3
        Conv1D(filters=48, activation='relu', kernel_size=3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # CNN Block 4 (Trước LSTM)
        Conv1D(filters=64, activation='relu', kernel_size=3),
        BatchNormalization(),
        # BiLSTM Layers
        Bidirectional(LSTM(128, return_sequences=True)),
        Activation('relu'),
        Bidirectional(LSTM(96)),
        # Fully Connected
        Dense(256, activation='relu', kernel_initializer='he_normal', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(units=num_classes, activation='softmax')
    ])
    return model

# Cấu hình
INPUT_SHAPE = (4096, 1)
NUM_CLASSES = 86

# A. Load Dữ liệu (Train/Valid/Test)
X_train, y_train = load_data_from_npz('train_data.npz', NUM_CLASSES)
X_val, y_val     = load_data_from_npz('valid_data.npz', NUM_CLASSES)
X_test, y_test   = load_data_from_npz('test_data.npz', NUM_CLASSES)

# B. Compile Model
model = my_model(INPUT_SHAPE, NUM_CLASSES)
optimizer = RMSprop(learning_rate=0.000173)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# C. Callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("\n>>> BẮT ĐẦU HUẤN LUYỆN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1,
    batch_size=256,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# D. Vẽ Loss Curve (Yêu cầu của bạn)
plt.figure(figsize=(12, 5))
# Biểu đồ Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# Biểu đồ Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Acc', color='red')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.show()
print(">> Đã lưu biểu đồ vào file 'loss_curve.png'")

# E. Đánh giá chi tiết để tái lập Table 3
model.load_weights('best_model.h5')

y_pred_prob = model.predict(X_test)
y_true_cls = np.argmax(y_test, axis=1)
y_pred_cls = np.argmax(y_pred_prob, axis=1)

# 1. Tính nhóm chỉ số MICRO (Chính là Global metrics)
accuracy = accuracy_score(y_true_cls, y_pred_cls)
micro_precision = precision_score(y_true_cls, y_pred_cls, average='micro')
micro_recall = recall_score(y_true_cls, y_pred_cls, average='micro')
micro_fscore = f1_score(y_true_cls, y_pred_cls, average='micro')

# 2. Tính nhóm chỉ số MACRO (Trung bình cộng của từng class)
precision_per_class = precision_score(y_true_cls, y_pred_cls, average=None, zero_division=0)
recall_per_class = recall_score(y_true_cls, y_pred_cls, average=None, zero_division=0)
f1_per_class = f1_score(y_true_cls, y_pred_cls, average=None, zero_division=0)

macro_precision = np.mean(precision_per_class)
macro_recall = np.mean(recall_per_class)
macro_fscore = np.mean(f1_per_class)

# 3. In bảng kết quả
print("\n" + "="*35)
print(f"{'Metric':<20} | {'Value (%)':<10}")
print("-" * 35)
print(f"{'Accuracy':<20} | {accuracy*100:6.2f}")
print(f"{'Micro-Precision':<20} | {micro_precision*100:6.2f}")
print(f"{'Micro-Recall':<20} | {micro_recall*100:6.2f}")
print(f"{'Micro-Fscore':<20} | {micro_fscore*100:6.2f}")
print("-" * 35)
print(f"{'Macro-Precision':<20} | {macro_precision*100:6.2f}")
print(f"{'Macro-Recall':<20} | {macro_recall*100:6.2f}")
print(f"{'Macro-Fscore':<20} | {macro_fscore*100:6.2f}")
print("="*35)