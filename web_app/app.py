from flask import *
import os
from werkzeug.utils import secure_filename
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)

with open('./model/trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

classes = {0:"Speed Limit 20 km/h",
          1:"Speed Limit 30 km/h",
          2:"Speed Limit 50 km/h",
          3:"Speed Limit 60 km/h",
          4:"Speed Limit 70 km/h",
          5:"Speed Limit 80 km/h",
          6:"End of Speed Limit 80 km/h",
          7:"Speed Limit 100 km/h",
          8:"Speed Limit 120 km/h",
          9:"No Passing",
          10:"No Passing Vehicle over 3.5 tons",
          11:"Right-of-way at Intersection",
          12:"Priority Road",
          13:"Yield",
          14:"Stop",
          15:"No Vehicles",
          16:"Vehicle > 3.5 tons Prohibited",
          17:"No Entry",
          18:"General Caution",
          19:"Dangerous Curve Left",
          20:"Dangerous Curve Right",
          21:"Double Curve",
          22:"Bumpy Road",
          23:"Slippery Road",
          24:"Road Narrows on the Right",
          25:"Road Work",
          26:"Traffic Signals",
          27:"Pedestrians Crossing",
          28:"Children Crossing",
          29:"Bicycles Crossing",
          30:"Beware of Ice/Snow",
          31:"Wild Animals Crossing",
          32:"End Speed + Passing Limits",
          33:"Turn Right Ahead",
          34:"Turn Left Ahead",
          35:"Ahead Only",
          36:"Go Straight or Right",
          37:"Go Straight or Left",
          38:"Keep Right",
          39:"Keep Left",
          40:"Roundabout Mandatory",
          41:"End of No Passing",
          42:"End of No Passing Vehicle over 3.5 tons"}

def get_prediction(image_path):
    d = []
    image = Image.open(image_path)
    image = image.resize((30,30))
    d.append(np.array(image))
    prob = model.predict(np.array(d))
    pred = np.argmax(prob)
    return pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = get_prediction(file_path)
        str = "Predicted Traffic🚦Sign is: " + classes[result]
        os.remove(file_path)
        return str
    return None

if __name__ == '__main__':
    app.run(debug=True)
    # pass