from flask import Flask,render_template,request, jsonify

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests

app = Flask(__name__,template_folder="../templates", 
            static_folder='../static')

model = load_model('nutrition.hdf5.h5')
print("Loaded model from disk")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image')
def image1():
    return render_template("image.html")

@app.route('/imageprediction')
def imageprediction():
    return render_template("imageprediction.html")


@app.route('/predict',methods=['POST'])
def launch():
    if request.method=='POST':
        f=request.files['file']
        
        basepath=os.path.dirname('/')
        filepath=os.path.join(basepath, f.filename)
        f.save(filepath)
        
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        
        pred=np.argmax(model.predict(x),axis=1)
        print("prediction",pred)
        index=['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON']
        
        result=str(index[pred[0]])
        apiResult=nutrition(result)
        
        final_result = {
                "result" : result, 
                "apiResult" : apiResult
             }
        print(final_result)
        return final_result
    
def nutrition(index1):
    
    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"

    querystring = {"query":"index1"}

    headers = {
	    "X-RapidAPI-Key": "46889440f7msh3ac5fd1bef5838ap1689ecjsna4a86408a0f5",
	    "X-RapidAPI-Host": "calorieninjas.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    
    print(response.text)
    
    return response.json()['items']

if __name__== "__main__":
    app.run(debug=False)
    
        
        
        