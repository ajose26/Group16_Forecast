from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List
import os

app = FastAPI()

# Load base data
historical_df = pd.read_csv("Food Prices.csv")
historical_df.columns = historical_df.columns.str.strip()
historical_df['date'] = pd.to_datetime(historical_df[['Year', 'Month']].assign(DAY=1))
historical_df = historical_df.sort_values('date')

class ForecastRequest(BaseModel):
    country: str
    item: str
    months: int

@app.post("/forecast/")
def get_forecast(req: ForecastRequest):
    model_name = f"{req.country.strip()}_{req.item.strip()}".replace(" ", "_")
    model_path = None

    for file in os.listdir("saved_models"):
        if model_name in file and file.endswith(".pkl"):
            model_path = os.path.join("saved_models", file)
            break

    if not model_path:
        return {"error": f"Model not found for {req.country} - {req.item}"}

    model = joblib.load(model_path)

    df = historical_df[
        (historical_df['Country'].str.strip() == req.country.strip()) &
        (historical_df['Food Item'].str.strip() == req.item.strip())
    ][['date', 'Price in USD']].set_index('date').sort_index()

    df_2022 = df[df.index.year == 2022]
    last_known = df_2022.iloc[-1]
    last_date = df_2022.index[-1]

    if "Prophet" in model_path:
        df = df.reset_index().rename(columns={'date': 'ds', 'Price in USD': 'y'})
        future = model.make_future_dataframe(periods=req.months, freq='MS')
        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].tail(req.months)
        result.columns = ['date', 'forecast']
    elif "SARIMA" in model_path:
        forecast_vals = model.forecast(steps=req.months)
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=req.months, freq='MS')
        result = pd.DataFrame({'date': dates, 'forecast': forecast_vals})
    elif "XGBoost" in model_path:
        series = df['Price in USD']
        lag1 = series[-1]
        lag2 = series[-2]
        preds = []
        for _ in range(req.months):
            pred = model.predict([[lag1, lag2]])[0]
            preds.append(pred)
            lag2, lag1 = lag1, pred
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=req.months, freq='MS')
        result = pd.DataFrame({'date': dates, 'forecast': preds})
    else:
        return {"error": "Unsupported model type."}

    return result.to_dict(orient="records")
