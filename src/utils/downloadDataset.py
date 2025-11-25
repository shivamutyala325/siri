from datasets import load_dataset


from datasets import load_dataset
import os

class downloadDataset():
    def __init__(self):
        self.datasetName = "naver-clova-ix/cord-v2"
        os.makedirs("../../data", exist_ok=True) 
        
    def download(self):
        
        dataset_dict = load_dataset(self.datasetName)
        dataset_dict["train"].save_to_disk("../../data/traindata")
        dataset_dict["test"].save_to_disk("../../data/testdata")
        dataset_dict["validation"].save_to_disk("../../data/validationdata")

        print("Dataset splits saved successfully to the respective folders in '../../data/'")


d = downloadDataset()
d.download()
