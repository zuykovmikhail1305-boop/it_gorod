def main():
    print("Hello from it-gorod!")


if __name__ == "__main__":
    main()


from lightgbm import LGBMRegressor

lightgbm_model = LGBMRegressor(
    objective="mape",
    n_estimators=10000,
    learning_rate=0.5,
    num_leaves=100,
    random_state=42,
    verbosity=-1
)

lightgbm_model.fit(X_train, y_train)
predictions_lightgbm = lightgbm_model.predict(X_test)

print(f"LightGBM MAE: {mean_absolute_error(y_test, predictions_lightgbm):.4f}")
print(f"LightGBM MAPE: {mape(y_test, predictions_lightgbm):.4f}%")