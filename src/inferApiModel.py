from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from google.genai import types

class useModel():
    def __init__(self):
        self.model='gemini-2.5-flash'
        self.prompt="You are a model for extracting the content from the invoice and return the list of items with quantity and price of each item"
        

    def generate(self):
        with open("../data/image1.png", 'rb') as f:
            image_bytes = f.read()

        client = genai.Client()
        response = client.models.generate_content(
            model=self.model,
            contents=[
            self.prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png',
            ),
            'Caption this image.'
            ]
        )

        print(response.text)

