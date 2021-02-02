# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request

model = pickle.load(open('aisclassifier.sav','rb'))
app = Flask(__name__)
meanVal = np.array([183.688783,183.000054,983.063901,4.100918])
stdDeviation = np.array([83.070587,82.326782,1.664243,1.454851])

#todo add weather data correction in the implementation

# def load_model():
#     global model
#     # model variable refers to the global variable
#     with open('aisclassifier.sav', 'rb') as f:
#         model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)

        # data normalization
        for i in range(4):
            data[0][i] = ((data[0][i]) - (meanVal[i]))/(stdDeviation[i])
            # print(data.shape)
            # print(str(i)+" "+str(data[0][i])+" "+str(meanVal[i])+" "+str(stdDeviation[i]))

        prediction = model.predict(data)  # runs globally loaded model on the data
        terr = (prediction[0]/15)
        terr = min(terr,1)
    return str(terr)

if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    # app.run(ssl_context='adhoc', host='127.0.0.1', port=5556)
    app.run()