import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from sklearn import preprocessing

# Function to calculate accuracy percentage
def calculate_accuracy(predictions):
    return np.max(predictions, axis=1) * 100

folder = r"C:\Users\Priyanshi Jain\Documents\images"

photos = []
for filename in os.listdir(folder):
    photo = load_img(os.path.join(folder, filename), target_size=(299, 299))
    photo = img_to_array(photo, dtype='uint8')
    photos.append(photo)

X = np.asarray(photos, dtype='uint8')

model = load_model(r"C:\Users\Priyanshi Jain\Documents\GitHub\Rajeshh\developed_model.h5")
yhats2 = model.predict(X)

max_predictions = (yhats2 == yhats2.max(axis=1, keepdims=1)).astype(int)

labels = np.array(['lung opacity', 'covid19', 'normal', 'pneumonia', 'pneumothorax'])
lb = preprocessing.LabelBinarizer()

y_labels = lb.fit_transform(labels)
y_prediction = lb.inverse_transform(max_predictions)

accuracies = calculate_accuracy(yhats2)

result = zip(y_prediction, accuracies)
for prediction, accuracy in result:
    print(f"Predicted_lung disease is {prediction} with {accuracy:.2f}% accuracy.")

DF = pd.DataFrame({'Predicted Disease': y_prediction, 'Accuracy (%)': accuracies})
DF.to_csv(r"C:\Users\Priyanshi Jain\Documents/prediction.csv", index=False)