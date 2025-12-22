import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, Dropout, BatchNormalization, MaxPooling1D, Dense, LSTM, 
                                     Bidirectional, Activation)
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score, average_precision_score, roc_curve, auc)
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def my_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        Dropout(0.1),
        Conv1D(filters=32, activation='relu', kernel_size=3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),
        Conv1D(filters=48, activation='relu', kernel_size=3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=64, activation='relu', kernel_size=3),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=True)),
        Activation('relu'),
        Bidirectional(LSTM(96)),
        Dense(256, activation='relu', kernel_initializer='he_normal', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(units=num_classes, activation='softmax')
    ])
    return model

# Assuming input_shape and num_classes are known
input_shape = (4096, 1)
num_classes = 86
model = my_model(input_shape, num_classes)
# Load the saved weights
weights_path = "weight.ckpt"
model.load_weights(weights_path)

# Compile the model
optimizer = RMSprop(learning_rate=0.000173)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# Load the test data
test_data_path = "fingerprints/test_LSTM.npz"
test_data = np.load(test_data_path)
X_test = test_data['X_test']
y_test = test_data['y_test']

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# Predict and evaluate the final trained model on the test set
y_pred_prob = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred_prob, axis=1)
import matplotlib.pyplot as plt

# Create empty lists to store metrics for each class
class_accuracy = []
class_auc = []
class_aupr = []
class_macro_precision = []
class_macro_recall = []
class_macro_fscore = []
class_micro_precision = []
class_micro_recall = []
class_micro_fscore = []
class_mcc = []

# Calculate metrics for each class
for i in range(num_classes):
    class_accuracy.append(accuracy_score(y_true[y_true == i], y_pred[y_true == i]))
    class_auc.append(roc_auc_score((y_true == i).astype(int), y_pred_prob[:, i]))
    class_aupr.append(average_precision_score((y_true == i).astype(int), y_pred_prob[:, i]))
    
    class_precision = precision_score((y_true == i).astype(int), (y_pred == i).astype(int))
    class_recall = recall_score((y_true == i).astype(int), (y_pred == i).astype(int))
    class_fscore = f1_score((y_true == i).astype(int), (y_pred == i).astype(int))
    class_mcc.append(matthews_corrcoef((y_true == i).astype(int), (y_pred == i).astype(int)))
    
    class_macro_precision.append(class_precision)
    class_macro_recall.append(class_recall)
    class_macro_fscore.append(class_fscore)
    
    class_micro_precision.append(class_precision)
    class_micro_recall.append(class_recall)
    class_micro_fscore.append(class_fscore)

# Combine the specific metrics into a single list
metrics = [
    class_accuracy, class_macro_precision, class_macro_recall, class_macro_fscore,
    class_micro_precision, class_micro_recall, class_micro_fscore,
    class_auc, class_aupr, class_mcc
]

# Define labels for the specific metrics
metric_labels = [
    'ACC', 'Macro-Precision', 'Macro-Recall', 'Macro-F1',
    'Micro-Precision', 'Micro-Recall', 'Micro-F1',
    'AUC', 'AUPRC', 'MCC'
]

# Create a list of different colors for each box
box_colors = ['blue', 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'green', 'magenta', 'lime']

# Plot a box plot for the specific metrics with different colors
plt.figure(figsize=(15, 10))
boxplot = plt.boxplot(metrics, labels=metric_labels, patch_artist=True, boxprops=dict(facecolor='white'))
for patch, color in zip(boxplot['boxes'], box_colors):
    patch.set_facecolor(color)

plt.title('Metrics Evaluation for all the Classes using BoxPlot')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Metric Value')
plt.ylim(0, 1)

# Hide the legend for clarity
plt.legend().set_visible(False)

plt.show()
plt.savefig('BoXPlot.png', dpi=400)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# ROC curve and AUC calculation
n_classes = 86
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for all classes and save to files
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
n_plots = n_classes // 10 + (1 if n_classes % 10 else 0)

for plot_index in range(n_plots):
    plt.figure(figsize=(10, 8))
    start_class = plot_index * 10
    end_class = min(start_class + 10, n_classes)

    for i, color in zip(range(start_class, end_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3, label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Specificity)', fontsize=20)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=20)
    plt.title(f'ROC for Classes {start_class} to {end_class - 1}', fontsize=16)
    plt.legend(loc="lower right", prop={'size': 10})
    file_name = f"visualization metrix/roc_curve_plot_{plot_index + 1}.png"
    plt.savefig(file_name, dpi=500)
    plt.show()
# Normalize and plot confusion matrix
cm_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(40, 40))
sns.heatmap(cm_percent, cmap='afmhot_r', square=True, cbar=True, linewidths=1, linecolor='black',
            xticklabels=np.arange(1, n_classes + 1), yticklabels=np.arange(1, n_classes + 1))

plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=24)
plt.title('Confusion Matrix Heatmap', fontsize=30)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
