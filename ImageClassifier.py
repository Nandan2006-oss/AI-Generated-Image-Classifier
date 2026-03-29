import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()

fake_folder = "test/fake"
real_folder = "test/real"

X = []
y = []

for file in os.listdir(fake_folder)[:100]:
    path = os.path.join(fake_folder, file)

    image = Image.open(path)
    image = image.resize((128, 128))
    image = image.convert("RGB")
    
    image_array = np.array(image)
    X.append(image_array.flatten())
    y.append(1)

for file in os.listdir(real_folder)[:100]:
    path = os.path.join(real_folder, file)

    image = Image.open(path)
    image = image.resize((128, 128))
    image = image.convert("RGB")
    
    image_array = np.array(image)
    X.append(image_array.flatten())
    y.append(0)

X = np.array(X)
y = np.array(y)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(X.shape)
print(X.min(), X.max())
print(len(X), len(y))
print(np.array(X).shape)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
print("Accuracy:", accuracy_score(y_test, y_pred))