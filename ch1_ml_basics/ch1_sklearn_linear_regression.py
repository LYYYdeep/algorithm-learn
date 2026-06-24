import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    
    y = np.array([3, 5, 7, 9, 11], dtype=float)
    
    model = LinearRegression()
    
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    

    print("Weight:", model.coef_)
    print("Bias:", model.intercept_)
    print("Train predictions:", y_pred)
    print("Train MSE:", mse)

    test_X = np.array([
        [6],
        [7],
    ], dtype=float)

    test_pred = model.predict(test_X)

    print("Test predictions:", test_pred)
    
if __name__ == "__main__":
    main()