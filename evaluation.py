import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn import svm

def calculateStatus(stock, sp500, outperformance=10):
    if outperformance >= 0:
        return stock - sp500 >= outperformance
    else:
        raise ValueError("outperformance must be positive")

def metricEvaluation():

    df = pd.read_csv("keystats.csv", index_col="Date")
    df.dropna(axis=0, how="any", inplace=True)

    features = df.columns[6:]
    X = df[features].values

    y = list(
        calculateStatus(
            df["stock_p_change"], df["SP500_p_change"], outperformance=10
        )
    )

    z = np.array(df[["stock_p_change", "SP500_p_change"]])

    X_tr, X_test, Y_tr, y_test, Z_tr, z_test = train_test_split(
        X, y, z, test_size=0.2
    )

    model = RandomForestClassifier(n_estimators=120, random_state=183)
    model.fit(X_tr, Y_tr)
    metricsForSpecificModel(model, X_test, y_test, z_test)

    model_svm = svm.SVC(kernel='linear')
    model_svm.fit(X_tr, Y_tr)
    metricsForSpecificModel(model_svm, X_test, y_test, z_test)


def metricsForSpecificModel(model, X_test, y_test,z_test):

    y_pred = model.predict(X_test)
    print("Performance of Classifier\n", "=" * 20)
    print(f"Accuracy : {model.score(X_test, y_test): .2f}")
    print(f"Precision : {precision_score(y_test, y_pred): .2f}")

    num_positive_predictions = sum(y_pred)
    if num_positive_predictions < 0:
        print("No stocks predicted!")

    stock_returns = 1 + z_test[y_pred, 0] / 100
    market_returns = 1 + z_test[y_pred, 1] / 100

    avg_stock_growth_predicted = sum(stock_returns) / num_positive_predictions
    index_growth = sum(market_returns) / num_positive_predictions
    percentage_stock_returns = 100 * (avg_stock_growth_predicted - 1)
    percentage_market_returns = 100 * (index_growth - 1)
    total_outperformance = percentage_stock_returns - percentage_market_returns

    print("\n  performance report of Stock prediction\n", "=" * 40)
    print(f"Total Trades:", num_positive_predictions)
    print(f"  stock predictions  average return: {percentage_stock_returns: .1f} %")
    print(
        f"Average market return : {percentage_market_returns: .1f}% "
    )
    print(
        f"Our technique outperforms the index by {total_outperformance: .1f} percentage points."



    )


if __name__ == "__main__":
    metricEvaluation()
