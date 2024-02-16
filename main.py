

import cv2
import numpy as np
import tensorflow as tf
#
# Laden des trainierten Modells (hier verwenden wir MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Definition der Klassenbezeichnungen für ImageNet
imagenet_classes = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(imagenet_classes) as f:
    imagenet_labels = f.readlines()

# Funktion zur Vorhersage und Klassifizierung eines Bildes
def predict_image(image):
    # Größenänderung und Normalisierung des Bildes entsprechend den Anforderungen des Modells
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Vorhersage des Bildes
    predictions = model.predict(np.expand_dims(image, axis=0))

    # Extraktion des vorhergesagten Indexes und des zugehörigen Labels
    predicted_index = np.argmax(predictions)
    label = imagenet_labels[predicted_index]

    return label

# Funktion zur Objekterkennung in einem Bild
def detect_objects(image_path):
    # Laden des Bildes
    image = cv2.imread(image_path)
    cv2.imshow("Bild", image)
    cv2.waitKey(0)
    
    # Umwandlung von BGR in RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Vorhersage und Klassifizierung des Bildes
    prediction = predict_image(image_rgb)
    
    return prediction

if __name__ == "__main__":
    # Beispielaufruf der Funktion zur Objekterkennung
    image_path = 'C:\\Users\\Hendrik\\Downloads\\Praktiukm_Java\\example.jpg'  # Passe den Dateipfad zu deinem Bild an

    prediction = detect_objects(image_path)
    print("Prediction:", prediction)
