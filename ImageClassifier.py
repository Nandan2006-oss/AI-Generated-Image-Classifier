import os
import numpy as np
from PIL import Image

fake_folder = "test/fake"
real_folder = "test/real"

X = []
y = []

for file in os.listdir(fake_folder)[1:5]:
    path = os.path.join(fake_folder, file)

    image = Image.open(path)
    image = image.resize((128, 128))
    image = image.convert("RGB")
    
    image_array = np.array(image)
    X.append(image_array.flatten())
    y.append(1)

    print(image.size)
    print(image.mode)

for file in os.listdir(real_folder)[1:5]:
    path = os.path.join(real_folder, file)

    image = Image.open(path)
    image = image.resize((128, 128))
    image = image.convert("RGB")
    
    image_array = np.array(image)
    X.append(image_array.flatten())
    y.append(0)

    print(image.size)
    print(image.mode)

print(len(X), len(y))
print(np.array(X).shape)
print(X)
print(y)