import os
import numpy as np
from PIL import Image

folder = "test/fake"

for file in os.listdir(folder)[1:5]:
    path = os.path.join(folder, file)

    image = Image.open(path)
    image = image.resize((128, 128))
    image = image.convert("RGB")
    
    image_array = np.array(image)

    print(image.size)
    print(image.mode)
