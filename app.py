from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline
import traceback 
application=Flask(__name__)

app=application
# route for a home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            data = CustomData(
                age=int(request.form.get('age')),
                sex=request.form.get('sex'),
                bmi=float(request.form.get('bmi')),
                children=int(request.form.get('children')),
                smoker=request.form.get('smoker'),
                region=request.form.get('region')
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)  # Debugging

            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('home.html', results=results[0])

    except Exception as e:
        error_message = traceback.format_exc()
        print("ERROR:", error_message)  # This will show full error logs
        return f"Internal Server Error: {error_message}", 500
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)