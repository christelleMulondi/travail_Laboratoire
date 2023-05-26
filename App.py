#Ce module fournit l'option DEBUG.
from distutils.log import debug
from flask import Flask,request,jsonify,render_template,redirect,url_for
import sklearn
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#chergement du model Adaptivity_Level.pkl
app=Flask(__name__)
models=pickle.load(open('Adaptivity_Level.pkl',"rb"))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    #chergement du model
    models=pickle.load(open('Adaptivity_Level.pkl',"rb"))
    #converssion des donnees
    int_futures=[float(i) for i in request.form.values()]
     #donner la forme aux donnees de mm maniere pytho
    dernier_futures=[np.array(int_futures)]
    # prise en charge de toutes les caracteristiques  
    dernier_futures=np.array([dernier_futures]).reshape(1,13)
    #Prediction en tenant compte de 3 elements de notre tareget
    predire=models.predict(dernier_futures)
    if(models.predict(dernier_futures)==0):
        predire="Moderate"
    elif (models.predict(dernier_futures)==1):
        predire="Low"
    else:
        predire="High"
    return render_template('index.html', prediction_text_=" Le resultat est : {}".format(predire))
    # debug permet de detecter l'erreur
if __name__=='__main__':
    app.run(debug=True)