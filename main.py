from typing import Annotated
from fastapi.responses import HTMLResponse
from fastapi import FastAPI,File,UploadFile
from fastapi.staticfiles import StaticFiles
import uuid
import os
import glob
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.models import load_model
from model import Model
import keras
# print(keras.__version__)
# print(tf.__version__)

app = FastAPI()

IMAGE_DIR = "./image/"
app.mount("/image", StaticFiles(directory=IMAGE_DIR), name="image")


@app.get("/")
def main():
    filenames = glob.glob("./image/*.jpg")
    # print(len(filenames))
    if len(filenames) > 0:
        os.remove(filenames[0])
        # print("f")
    content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .container { width: 50%; margin: auto; padding: 20px; }
            h1 { text-align: center; }
            form { display: flex; flex-direction: column; align-items: center; }
            input[type="file"] { margin-bottom: 10px; }
            input[type="submit"] { padding: 5px 15px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload an Image for Prediction</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Predict">
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content)


@app.post("/predict/")
async def get_predictions(file: UploadFile = File(...)):
 
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
 
    #save the file
    with open(f"{IMAGE_DIR}{file.filename}", "wb") as f:
        f.write(contents)
    
    model = Model()
    pred = model.getPrediction(f"{IMAGE_DIR}{file.filename}")
    html_content = f"""
    <html>
    <body>
        <h2>Uploaded Image</h2>
        <img src='/image/{file.filename}' width='300'/>
        <h2>Prediction Result</h2>
        <p>{pred}</p>
        <br>
        <a href="/">Go Back</a>
    </body>
    </html>
    """
    # os.remove(f"{IMAGE_DIR}{file.filename}")
    return HTMLResponse(content=html_content)
