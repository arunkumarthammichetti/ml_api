
import pandas as pd
from datetime import datetime
from datetime import timedelta

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from statsmodels.tsa.arima_model import ARIMA


app = Flask(__name__)
CORS(app)







@app.route('/')
def home():
    return 'ARIMA_Analysis'


@app.route('/predict')
def predict():
    df = pd.read_csv("AAPL.csv")
    df = df[['Date', 'Open']]
    pdf = df['Open']
    fdf = pd.DataFrame(columns=['Open'])
    history = [x for x in df['Open']]
    predictions = list()
    for t in range(len(pdf) + 60):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = pdf[t]
        history.append(obs)
        if (t >= len(pdf) - 1):
            pdf = pdf.append(pd.Series([yhat, yhat], index=df.columns), ignore_index=True)
            fdf = fdf.append(pd.Series(yhat, index=fdf.columns), ignore_index=True)
    rdf = pd.DataFrame(columns=['Date'])
    Date = datetime.strptime(df[-1:]['Date'].tolist()[0], "%Y-%m-%d")
    rdf = rdf.append(pd.Series((Date + timedelta(days=1)).strftime("%Y-%m-%d"), index=rdf.columns), ignore_index=True)
    for i in range(1, 30):
        Date = datetime.strptime(rdf[-1:]['Date'].tolist()[0], "%Y-%m-%d")
        rdf = rdf.append(pd.Series((Date + timedelta(days=1)).strftime("%Y-%m-%d"), index=rdf.columns), ignore_index=True)
    apidf = pd.concat([rdf, fdf[:-1]], axis=1, join='inner')



    return str(apidf.set_index('Date').T.to_dict('records'))




if __name__ == '__main__':
    app.run(debug=True)