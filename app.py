# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request

model = pickle.load(open('aisclassifier.sav','rb'))
app = Flask(__name__)
meanVal = np.array([183.688783,183.000054,983.063901,4.100918])
stdDeviation = np.array([83.070587,82.326782,1.664243,1.454851])
shipclassDWT = np.array([300705,120920,54106,43439])
length = np.array([330,220,180])
breadth = np.array([60,40,28])
power = np.array([36419,20063,13490,12118])

# >330 ULCC + VLCC
# >220 LR2 ( 80,000 - 159,999)
# >180 LR1 (45,000,79,999)
# ~rest ( medium rannge)
#todo add weather data correction in the implementation

# Fomula for estimation :
#      ESTIMATION VALUE = (Pv * Carbon conversion factor
#                       ((vc/vpredicted)^3)*Tm)/ 
                        # (cargo carried* distance travel)
#  arguments : 

#  Pv           : Power delivery of the ship's engine
#  vc           : Velocity of the ship
#  vpredicted   : Ideal Velocity of the ship
#  Tm           : Time duration of execution
#  Cm           : Cargo carried by the vessel
#  D            : distance travel


@app.route('/')
def home_endpoint():
    return 'Hello World!'


def EEDICALC(predicted_Velocity, EEDIvars):
    shiplength   = EEDIvars[0]
    shipbreadth  = EEDIvars[1]
    shipcargo    = EEDIvars[2]/1000
    shipdistance = EEDIvars[3]/1.852
    shipvelocity = EEDIvars[4]
    vesselClass  = 3
    while(vesselClass>0):
       if(shiplength<length[vesselClass-1]):
           vesselClass= vesselClass-1
       else: 
           break

    vesselPower = power[vesselClass]
    velocityRatio = (shipvelocity/predicted_Velocity)**3
    emissionFactor = 0.670
    cutoffValue = 0.8*4000

    numerator = vesselPower*velocityRatio*(shipdistance/shipvelocity)*emissionFactor
    denominator = shipcargo*shipdistance
    EEDI = numerator/denominator
    return (cutoffValue-EEDI)/cutoffValue
    
@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        tempdata = request.get_json()  # Get data posted as a json
        data = tempdata[0:4]
        EEDIvars = tempdata[4:]
       
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)

        # recieved parameters
        # COG, Heading, Pressure, Temperature, Lenth, Breadth, Cargo, Diatance,Speed of vessel

        for i in range(4):
            data[0][i] = ((data[0][i]) - (meanVal[i]))/(stdDeviation[i])
            # print(data.shape)
            # print(str(i)+" "+str(data[0][i])+" "+str(meanVal[i])+" "+str(stdDeviation[i]))

        prediction = model.predict(data)  # runs globally loaded model on the data
        terr = EEDICALC(prediction[0],EEDIvars)
        print(terr)
        # terr = (prediction[0]/15)
    
        terr = min(terr,1)
    return str(terr)

if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(ssl_context='adhoc', host='127.0.0.1', port=5556)
    # app.run()