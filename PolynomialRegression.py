from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

if __name__ == '__main__':
    # Reading input from user
    F, N = map(int, input().split())

    X_train = []
    y_train = []

    for _ in range(N):
        train_data = list(map(float, input().split()))
        X_train.append(train_data[:-1])
        y_train.append(train_data[-1])

    T = int(input())

    X_test = []

    for _ in range(T):
        test_data = list(map(float, input().split()))
        X_test.append(test_data[:])

    # Transform X into X_train_poly
    poly_reg = PolynomialFeatures(degree=3)
    X_train_poly = poly_reg.fit_transform(X_train)


    # Training
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train)

    # Transform X_test into X_test_poly
    X_test_poly = poly_reg.fit_transform(X_test)

    # Predicting
    y_preds = lin_reg.predict(X_test_poly)

    for y_pred in y_preds:
        print(round(y_pred, 2))
