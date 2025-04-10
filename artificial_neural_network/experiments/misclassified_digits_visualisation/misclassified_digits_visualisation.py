import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load training and testing datasets
train_data = pd.read_csv('../dataset_assets/optdigits.tra', header=None)
test_data = pd.read_csv('../dataset_assets/optdigits.tes', header=None)


# Split features and labels
X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Normalize pixel values
X_train = X_train / 16.0  # values range from 0–16
X_test = X_test / 16.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(64,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f'\nTest Accuracy: {test_acc:.4f}')
print(f'\nTest Loss: {test_loss:.4f}')


# Save accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/accuracy_plot.png')
plt.close()

# Save loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/loss_plot.png')
plt.close()

# Confusion matrix
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')
plt.close()

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

# Find misclassified indices
misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]

# Filter: Specific confusion types
mask_3_8 = (y_true_classes == 3) & (y_pred_classes == 8)
mask_7_9 = (y_true_classes == 7) & (y_pred_classes == 9)
mask_8_1 = (y_true_classes == 8) & (y_pred_classes == 1)
mask_8_9 = (y_true_classes == 8) & (y_pred_classes == 9)

indices_3_8 = np.where(mask_3_8)[0]
indices_7_9 = np.where(mask_7_9)[0]
indices_8_1 = np.where(mask_8_1)[0]
indices_8_9 = np.where(mask_8_9)[0]

# Plot function
def plot_all_misclassifications(pairs, n=5):
    rows = len(pairs)
    plt.figure(figsize=(n * 2.2, rows * 2.5))  # Wider and taller

    for row_idx, (indices, title) in enumerate(pairs):
        for i, idx in enumerate(indices[:n]):
            img = X_test_np[int(idx)].reshape(8, 8)
            true_label = y_true_classes[int(idx)]
            pred_label = y_pred_classes[int(idx)]

            plot_idx = row_idx * n + i + 1
            plt.subplot(rows, n, plot_idx)
            plt.imshow(img, cmap='gray')
            plt.title(f'True: {true_label}\nPred: {pred_label}', fontsize=9)
            plt.axis('off')

    plt.suptitle("Key Misclassifications", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the main title
    plt.show()



# Show key pairs
X_test_np = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test
pairs_to_plot = [
    (indices_3_8, "3 vs 8"),
    (indices_7_9, "7 vs 9"),
    (indices_8_1, "8 vs 1"),
    (indices_8_9, "8 vs 9")
]

plot_all_misclassifications(pairs_to_plot, n=5)
