import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# This is the percentage by which a stock has to beat S&P500 to be considered a 'buy'
OUTPERFORMANCE = 10


def convertStringToFloat(number_string):
    if ("N/A" in number_string) or ("NaN" in number_string):
        return "N/A"
    elif number_string == ">0":
        return 0
    elif "K" in number_string:
        return float(number_string.replace("K", "")) * 1000
    elif "B" in number_string:
        return float(number_string.replace("B", "")) * 1000000000
    elif "M" in number_string:
        return float(number_string.replace("M", "")) * 1000000
    else:
        return float(number_string)


def calculateStatus(stock, sp500, outperformance=10):
    if outperformance >= 0:
        return stock - sp500 >= outperformance
    else:
        raise ValueError("outperformance must be positive")


def datasetBuilder():
    training_data = pd.read_csv("keystats.csv", index_col="Date")
    training_data.dropna(axis=0, how="any", inplace=True)
    features = training_data.columns[6:]

    X_tr = training_data[features].values
    y_tr = list(
        calculateStatus(
            training_data["stock_p_change"],
            training_data["SP500_p_change"],
            OUTPERFORMANCE,
        )
    )
    return X_tr, y_tr


def stockPredictor():
    X_tr, y_tr = datasetBuilder()
    model = RandomForestClassifier(n_estimators=120, random_state=183)
    model.fit(X_tr, y_tr)

    data = pd.read_csv("forward_sample.csv", index_col="Date")
    data.dropna(axis=0, how="any", inplace=True)
    features = data.columns[6:]
    X_test = data[features].values
    z = data["Ticker"].values

    y_pr = model.predict(X_test)
    if sum(y_pr) == 0:
        print("No stocks are predicted!")
    else:
        print(
            f"{len(z[y_pr].tolist())} stocks predicted to outperform the S&P500 by more than {OUTPERFORMANCE}%:"
        )
        print(" ".join(z[y_pr].tolist()))
        return z[y_pr].tolist()


if __name__ == "__main__":
    stockPredictor()