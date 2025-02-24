import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Example data: replace this with your handover data
data = pd.Series([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                  115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])

# Fit the ARIMA model
model = ARIMA(data, order=(5, 1, 0))  # Example parameters
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecasting
forecast = model_fit.forecast(steps=5)  # Predicting the next 5 points
print(forecast)

# Plotting the data and forecast
plt.plot(data, label='Original')
plt.plot(np.arange(len(data), len(data) + len(forecast)), forecast, label='Forecast')
plt.legend()
plt.show()
