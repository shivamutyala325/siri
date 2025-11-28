from datasets import load_dataset, Dataset
from pathlib import Path
import cv2 as cv2
from PIL import Image
import numpy as np

class loadData:
    def __init__(self):
        script_dir = Path(__file__).parent
        base_dir = script_dir / "../../data"
        
        self.train_file = str(base_dir / "traindata/data-00000-of-00004.arrow")
        self.test_file = str(base_dir / "testdata/data-00000-of-00001.arrow")
        self.validation_file = str(base_dir / "validationdata/data-00000-of-00001.arrow")


    def loadTestdata(self):
        return load_dataset(
            'arrow', 
            data_files={'test': self.test_file}
        )

        
    
    def loadTraindata(self):
        return load_dataset(
            'arrow', 
            data_files={'train': self.train_file}
        )
        

    def loadValidationdata(self):
        return load_dataset(
            'arrow', 
            data_files={'validation': self.validation_file}
        )
        


d=loadData()
traindata=d.loadTraindata()
sample=traindata["train"][1]
sampleImage=np.array(sample["image"])
sampleImage=cv2.resize(sampleImage,(500,500))
samplegroundTruth=sample["ground_truth"]
print(samplegroundTruth)

cv2.imshow("sampleImage",sampleImage)

cv2.waitKey(0)
