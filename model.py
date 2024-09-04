from keras.models import load_model
import cv2
import numpy as np
import gdown
class Model:
    def download_model():
        url = "https://drive.google.com/uc?export=download&id=18cPcOnLm3sTy3lqXQGapKzR2xH6XMAfY"
        output = "finetuned.keras"
        gdown.download(url,output,quiet=True)

    def __init__(self):
        self.k_model = load_model('/finetuned.keras')
    
    def getPrediction(self,filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        img = np.array(img)
        img = img.reshape(1,128,128,3)
        pred = self.k_model.predict(img)
        is_dog= pred[0][0] > 0.7
        is_cat = pred[0][1] > 0.7
        temp = ""
        if is_dog:
            temp = "Its a Dog"
        elif is_cat:
            temp = "its a cat"
        else:
            temp = "neighter cat nor a dog"

        return temp

