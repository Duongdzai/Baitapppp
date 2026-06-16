import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from dataset_helper import load_custom_dataset

# 1. Load & Preprocess Custom Dataset
print("Dang tai du lieu tu custom_dataset...")
X, y, class_names = load_custom_dataset(base_path='custom_dataset', img_size=(28, 28), color_mode='grayscale')

# Flatten anh (28x28 ve 784 phan tu)
X_flat = X.reshape(X.shape[0], -1)

# Chia tap Train/Test (ty le 80/20)
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# 2. Train Perceptron
print("Dang huan luyen Perceptron...")
clf = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluate
y_pred = clf.predict(X_test)
print("\nBao cao chi tiet phan lop:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 4. Plot Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues')
plt.title("Ma tran nham lan (Perceptron) - Custom Dataset")
plt.show()
