import pandas as pd
import re
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np

class LabelExtractor:
    def __init__(self):
        self.extracted_text = None
        self.df_transposed = None

    def extract_nutritional_info(self, extracted_text):
        """
        Extracts nutritional information from the extracted text using regex patterns.

        Parameters:
            extracted_text (str): The OCR extracted text containing nutritional information.

        Returns:
            pd.DataFrame: A DataFrame with nutrients, their amounts, and % Daily Values.
        """
        
        # Define regex patterns for each nutritional item
        patterns = {
            "servings_container": r"(\d+)\s*servings\s*per\s*container|Servings:\s*(\d+)",
            "serving_size": r"Serving\s*size\s*([^\n%]+?\))|Serv.\s*size:\s*([^\n%]+?\))",
            "calories": r"Calories\s*(\d+)",
            "total_fat": r"Total\s*Fat\s*(\d+\.?\d*g?)",
            "saturated_fat": r"Saturated\s*Fat\s*(\d+\.?\d*g?)",
            "trans_fat": r"Trans\s*Fat\s*(\d+\.?\d*g?)",
            "cholesterol": r"Cholesterol\s*(\d+mg\s*\d*%?)",
            "sodium": r"Sodium\s*(\d+mg\s*\d*%?)",
            "total_carbohydrate": r"Total\s*(?:Carbohydrate|Carb\.)\s*(\d+g\s*\d*%)",
            "dietary_fiber": r"Dietary\s*Fiber\s*(\d+g\s*\d*%)|Fiber\s*(\d+g\s*\d*%)",
            "total_sugars": r"Total\s*Sugars\s*(\d+g)",
            "protein": r"Protein\s*(\d+g)"
        }

        # Extract data using regex patterns
        data = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, extracted_text)
            data[key] = next((m for m in match.groups() if m), None) if match else None

        # Convert extracted data to a DataFrame
        df = pd.DataFrame([data])

        # Transpose the DataFrame for better readability
        df_transposed = df.T.reset_index()
        df_transposed.columns = ['Nutrient', 'Value']

        # Split values into 'Weight' and 'DV%' columns
        df_transposed[['Weight', 'DV%']] = df_transposed['Value'].str.extract(r'(\d+\.?\d*(?:mg|g|mcg)?)\s*(\d+%|\d+% DV)?')

        self.df_transposed = df_transposed
        return df_transposed

    def show_table(self):
        """
        Displays the DataFrame with extracted nutritional information.
        """
        if self.df_transposed is not None:
            print("Extracted Nutritional Information:")
            display(self.df_transposed)
        else:
            print("No data to display. Please run `extract_nutritional_info` first.")



    def process_label(self, image_file_name):
        """
        Processes the image file, performs OCR to extract text, and displays the nutritional information.

        Parameters:
            image_file_name (str): The path to the image file to process.
        """
        # Open the image
        with open(image_file_name, 'rb') as f:
            image = Image.open(io.BytesIO(f.read()))

        # Convert the image to grayscale
        open_cv_image = np.array(image.convert('L'))  # Convert PIL image to grayscale

        # Apply thresholding
        _, thresh_image = cv2.threshold(open_cv_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL image for OCR
        preprocessed_image = Image.fromarray(thresh_image)

        # Perform OCR to extract text
        self.extracted_text = pytesseract.image_to_string(preprocessed_image)

        # Extract and show the nutritional information table
        table = self.extract_nutritional_info(self.extracted_text)
        self.show_table()


