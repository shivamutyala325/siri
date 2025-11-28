from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
from pathlib import Path



class getModel:
    def __init__(self):
        self.model_name="impira/layoutlm-invoices"
        self.model_path = Path(__file__).parent / "../../artifacts"


    def download(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        
        self.tokenizer.save_pretrained(self.model_path)
        self.model.save_pretrained(self.model_path)
        print(f"model saved sucessfully at {self.model_path}")


d=getModel()
d.download()