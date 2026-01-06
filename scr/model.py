from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def get_model(model_type="linear"):
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type")
