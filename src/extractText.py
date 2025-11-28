import pytesseract
from pytesseract import Output
from PIL import Image
from pathlib import Path



class GetText:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.model_path = (Path(__file__).parent / "../artifacts").resolve().as_posix()

    def get(self):
        img = Image.open(r"C:\Users\shiva\Desktop\SiriProject\siri\data\image2.png")
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        print(ocr_data)

d = GetText()
d.get()
