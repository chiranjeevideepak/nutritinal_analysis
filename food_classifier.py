import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from google.colab import files
import cv2

class FoodClassifier:
    def __init__(self, model_path, class_names):
        """
        Initialize the FoodClassifier with a trained model and class names.
        
        Parameters:
        - model_path (str): Path to the trained model file.
        - class_names (list): List of class names for predictions.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.uploaded_image_path = None
        self.class_name = None  # Store predicted class name for API usage
        self.max_weight = None  # Store estimated weight for API usage

    def upload_image(self):
        uploaded = files.upload()
        self.uploaded_image_path = next(iter(uploaded))
        print(f"Uploaded image: {self.uploaded_image_path}")

    def preprocess_image(self, img_path, img_height=128, img_width=128):
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array

    def predict_image(self):
        if self.uploaded_image_path is None:
            print("Please upload an image first.")
            return

        img_array = self.preprocess_image(self.uploaded_image_path)
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        self.class_name = self.class_names[predicted_class]  # Store the predicted class name

        plt.imshow(image.load_img(self.uploaded_image_path))
        plt.axis('off')
        plt.show()
        print(f"Predicted Class: {self.class_name}")

    def estimate_weight(self, pixel_to_cm_ratio=0.1, height_cm=2, density=0.9):
        if self.uploaded_image_path is None:
            print("Please upload an image first.")
            return
        
        image = cv2.imread(self.uploaded_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_weight = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            area_cm2 = area * (pixel_to_cm_ratio ** 2)
            volume_cm3 = area_cm2 * height_cm
            weight_g = volume_cm3 * density
            if weight_g > max_weight:
                max_weight = weight_g

        self.max_weight = max_weight
        print(f"Estimated Weight: {self.max_weight:.2f} grams")

    def get_nutritional_facts(self, food_name=None, quantity_in_grams=None):
        food_name = food_name or self.class_name
        quantity_in_grams = quantity_in_grams or self.max_weight

        if food_name is None or quantity_in_grams is None:
            print("Please provide both food name and quantity.")
            return

        search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={food_name}&search_simple=1&action=process&json=1"
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        try:
            response = session.get(search_url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Open Food Facts API: {str(e)}")
            return

        if data.get('count', 0) == 0:
            print(f"No results found for {food_name}.")
            return

        product = data['products'][0]
        nutriments = product.get('nutriments', {})

        if not nutriments:
            print(f"No nutritional data available for {food_name}.")
            return

        scale_factor = quantity_in_grams / 100
        nutritional_facts = {
        'Calories (kcal)': round(nutriments.get('energy_100g', 0) * scale_factor, 2),
        'Fat (g)': round(nutriments.get('fat_100g', 0) * scale_factor, 2),
        'Protein (g)': round(nutriments.get('proteins_100g', 0) * scale_factor, 2),
        'Carbohydrates (g)': round(nutriments.get('carbohydrates_100g', 0) * scale_factor, 2),
        'Fiber (g)': round(nutriments.get('fiber_100g', 0) * scale_factor, 2),
        'Sugars (g)': round(nutriments.get('sugars_100g', 0) * scale_factor, 2),
        'Sodium (mg)': round(nutriments.get('sodium_100g', 0) * scale_factor, 2),
    }

        return nutritional_facts
