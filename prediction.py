import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("blood_group_model.h5")

classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

img = cv2.imread("test.jpg")
img = cv2.resize(img, (224,224))
img = img/255.0
img = np.reshape(img, (1,224,224,3))

prediction = model.predict(img)
print("Predicted Blood Group:", classes[np.argmax(prediction)])