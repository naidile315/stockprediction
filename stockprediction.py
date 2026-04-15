
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path = "NIFTY50/RELIANCE.csv"
data = pd.read_csv(file_path)


print("First 5 Rows of Dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())


data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')


data = data[['Date', 'Close']]


forecast_days = 30
data['Prediction'] = data['Close'].shift(-forecast_days)


X = np.array(data[['Close']])[:-forecast_days]
y = np.array(data['Prediction'])[:-forecast_days]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

future_X = np.array(data[['Close']])[-forecast_days:]
predictions = model.predict(future_X)

print("\nPredicted Stock Prices for the Next 30 Days:")
for i, price in enumerate(predictions, start=1):
    print(f"Day {i}: ₹{price:.2f}")


plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label="Historical Prices")

future_dates = pd.date_range(
    start=data['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

plt.plot(future_dates, predictions, label="Predicted Prices", linestyle='dashed')

plt.title("NIFTY 50 Stock Price Prediction (RELIANCE)")
plt.xlabel("Date")
plt.ylabel("Stock Price (₹)")
plt.legend()
plt.grid()
plt.show()
