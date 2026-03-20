from pathlib import Path
import pandas as pd
import numpy as np

DOCS = Path("docs")
OUTPUT_FILE = DOCS / "final_climate_dataset.csv"
BASELINE_YEAR = 1958



# Load / clean sources

#clean nasa info first
def clean_nasa_temp(path):
    df = pd.read_csv(path, skiprows=1, na_values=["***"])

    # clean column names
    df.columns = [str(c).strip() for c in df.columns]

    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    keep_cols = ["Year"] + [c for c in month_cols if c in df.columns]
    df = df[keep_cols].copy()

    # convert to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    for col in month_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    df = df.melt(id_vars="Year", var_name="Month", value_name="Temp")

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    df["MonthNum"] = df["Month"].map(month_map)
    df = df.dropna(subset=["MonthNum", "Temp"])

    df["Date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["MonthNum"].astype(int), day=1)
    )

    return df[["Date", "Temp"]].sort_values("Date").reset_index(drop=True)

# clean noaa info dataset
def clean_noaa_gas(path, value_name):
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["average"] = pd.to_numeric(df["average"], errors="coerce")

    df = df.dropna(subset=["year", "month", "average"])
    df = df[df["average"] > 0]

    df["Date"] = pd.to_datetime(
        dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1)
    )

    df = df.rename(columns={"average": value_name})
    return df[["Date", value_name]].sort_values("Date").reset_index(drop=True)

# clean our world in data dataset using world data
def clean_owid(path, location="World"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    entity_col = "location" if "location" in df.columns else "country"
    df = df[df[entity_col] == location].copy()

    keep_cols = [
        "year",
        "primary_energy_consumption",
        "land_use_change_co2",
        "cement_co2",
        "flaring_co2",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    #rename for consistency
    rename_map = {
        "year": "Year",
        "primary_energy_consumption": "PrimaryEnergy",
        "land_use_change_co2": "LandUseCO2",
        "cement_co2": "CementCO2",
        "flaring_co2": "FlaringCO2",
    }
    df = df.rename(columns=rename_map)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    return df.sort_values("Year").reset_index(drop=True)



# Merge into one dataset
def merge_raw_data():
    temp = clean_nasa_temp(DOCS / "GLB.Ts+dSST.csv")
    co2 = clean_noaa_gas(DOCS / "co2_mm_gl.csv", "CO2")
    ch4 = clean_noaa_gas(DOCS / "ch4_mm_gl.csv", "CH4")
    n2o = clean_noaa_gas(DOCS / "n2o_mm_gl.csv", "N2O")
    owid = clean_owid(DOCS / "owid-co2-data.csv", location="World")

    # these prints were suggested by chatGPT
    print("temp:", temp.shape, temp["Date"].min(), temp["Date"].max())
    print("co2 :", co2.shape, co2["Date"].min(), co2["Date"].max())
    print("ch4 :", ch4.shape, ch4["Date"].min(), ch4["Date"].max())
    print("n2o :", n2o.shape, n2o["Date"].min(), n2o["Date"].max())
    print("owid:", owid.shape, owid["Year"].min(), owid["Year"].max())

    merged = temp.merge(co2, on="Date", how="inner")
    print("after temp+co2:", merged.shape)

    merged = merged.merge(ch4, on="Date", how="inner")
    print("after +ch4:", merged.shape)

    merged = merged.merge(n2o, on="Date", how="inner")
    print("after +n2o:", merged.shape)

    merged["Year"] = merged["Date"].dt.year
    merged = merged.merge(owid, on="Year", how="left")
    print("after +owid:", merged.shape)

    #show any missing data after the merge. will try to fix in preprocessing
    print("\nMissing values after merge:")
    print(merged.isna().sum())

    return merged.sort_values("Date").reset_index(drop=True)


# Data preprocessing
def preprocess_data(df):
    df = df.copy()

    # make sure Date is monthly and sorted
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # convert numeric columns
    numeric_cols = [c for c in df.columns if c != "Date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    owid_cols = ["PrimaryEnergy", "LandUseCO2", "CementCO2", "FlaringCO2"]
    owid_cols = [c for c in owid_cols if c in df.columns]
    df[owid_cols] = df[owid_cols].ffill().bfill()

    # interpolate the core monthly climate columns
    core_cols = [c for c in ["Temp", "CO2", "CH4", "N2O"] if c in df.columns]
    df[core_cols] = df[core_cols].interpolate(method="linear", limit_direction="both")

    # drop rows only if core climate variables are missing
    df = df.dropna(subset=core_cols)

    # feature engineering
    df["TimeSinceBaseline"] = (
        (df["Date"].dt.year - BASELINE_YEAR) * 12 + (df["Date"].dt.month - 1)
    )

    # sine and cosine encoding to capture seasonality as a cycle. ChatGPT helped with calculations
    df["MonthSin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)

    if "CO2" in df.columns:
        df["CO2_GrowthRate"] = df["CO2"].pct_change().fillna(0) * 100

    if "CH4" in df.columns:
        df["CH4_GrowthRate"] = df["CH4"].pct_change().fillna(0) * 100

    if "N2O" in df.columns:
        df["N2O_GrowthRate"] = df["N2O"].pct_change().fillna(0) * 100

    # 12-month moving average for each gas
    for gas in ["CO2", "CH4", "N2O"]:
        if gas in df.columns:
            df[f"{gas}_MA12"] = df[gas].rolling(12, min_periods=1).mean()

    if "Temp" in df.columns:
        df["Temp_MA12"] = df["Temp"].rolling(window=12, min_periods=1).mean()

    # lagged temperature values 1 month and 12 months
    df["Temp_Lag1"] = df["Temp"].shift(1).bfill()
    df["Temp_Lag12"] = df["Temp"].shift(12).bfill()

    # second derivative of CO2 to capture acceleration in growth. this was suggested by chatGPT
    df["CO2_Accel"] = df["CO2"].diff().diff().fillna(0)

    keep_cols = [
        "Date",
        "Temp",
        "CO2",
        "CH4",
        "N2O",
        "PrimaryEnergy",
        "LandUseCO2",
        "CementCO2",
        "FlaringCO2",
        "TimeSinceBaseline",
        "MonthSin",
        "MonthCos",
        "CO2_GrowthRate",
        "CH4_GrowthRate",
        "N2O_GrowthRate",
        "CO2_MA12",
        "CH4_MA12",
        "N2O_MA12",
        "Temp_MA12",
        "Temp_Lag1",
        "Temp_Lag12",
        "CO2_Accel",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


def main():
    merged = merge_raw_data()
    final_df = preprocess_data(merged)

    print("Final shape:", final_df.shape)
    print(final_df.head())
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # set up X and y, drop date and target columns
    X = final_df.drop(columns=["Date", "Temp", "Temp_MA12"])
    y = final_df["Temp"]

    # use shuffle=False to keep time order intact for the split
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )

    # scale after split to avoid data leakage. I was getting significant data issues before using the scalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sk)
    X_test_scaled = scaler.transform(X_test_sk)

    # Model 1: Linear Regression
    regressor_sk = LinearRegression()
    regressor_sk.fit(X_train_scaled, y_train_sk)

    y_pred_lr = regressor_sk.predict(X_test_scaled)

    print("Linear Regression")
    print("Train R-squared:", regressor_sk.score(X_train_scaled, y_train_sk))
    print("Test R-squared:", regressor_sk.score(X_test_scaled, y_test_sk))
    print("RMSE:", np.sqrt(mean_squared_error(y_test_sk, y_pred_lr)))
    print("MAE:", mean_absolute_error(y_test_sk, y_pred_lr))

    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train_sk, y_train_sk)

    y_pred_rf = rf.predict(X_test_sk)

    print("Random Forest")
    print("Train R-squared:", rf.score(X_train_sk, y_train_sk))
    print("Test R-squared:", rf.score(X_test_sk, y_test_sk))
    print("RMSE:", np.sqrt(mean_squared_error(y_test_sk, y_pred_rf)))
    print("MAE:", mean_absolute_error(y_test_sk, y_pred_rf))

    # Model interpretation

    coeff_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": regressor_sk.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    print(coeff_df)

    rf_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    # Figure 1: Time series of greenhouse gases vs temperature
    df_plot = final_df.copy()

    for col in ["Temp", "CO2", "CH4", "N2O"]:
        if col in df_plot.columns:
            df_plot[col] = (df_plot[col] - df_plot[col].mean()) / df_plot[col].std()

    plt.figure(figsize=(12, 6))
    plt.plot(df_plot["Date"], df_plot["Temp"], label="Temperature", linewidth=2)

    if "CO2" in df_plot.columns:
        plt.plot(df_plot["Date"], df_plot["CO2"], label="CO2")
    if "CH4" in df_plot.columns:
        plt.plot(df_plot["Date"], df_plot["CH4"], label="CH4")
    if "N2O" in df_plot.columns:
        plt.plot(df_plot["Date"], df_plot["N2O"], label="N2O")

    plt.title("Time Series: Greenhouse Gases vs Temperature")
    plt.xlabel("Date")
    plt.ylabel("Standardized Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(DOCS / "fig1_timeseries.png", dpi=150)
    plt.show()

    # Figure 2: Linear Regression feature importance
    coeff_plot = pd.DataFrame({
        "Feature": X.columns,
        "AbsCoefficient": np.abs(regressor_sk.coef_)
    }).sort_values(by="AbsCoefficient", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(coeff_plot["Feature"], coeff_plot["AbsCoefficient"])
    plt.title("Feature Importance (Linear Regression)")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(DOCS / "fig2_lr_feature_importance.png", dpi=150)
    plt.show()

    # Figure 3: Random Forest feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(rf_importance["Feature"], rf_importance["Importance"])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(DOCS / "fig3_rf_feature_importance.png", dpi=150)
    plt.show()

    # Figure 4: Scatter plot of top RF feature vs temperature
    top_feature = rf_importance.sort_values("Importance", ascending=False).iloc[0]["Feature"]

    plt.figure(figsize=(6, 5))
    plt.scatter(final_df[top_feature], final_df["Temp"], alpha=0.6)
    plt.title(f"{top_feature} vs Temperature")
    plt.xlabel(top_feature)
    plt.ylabel("Temperature Anomaly")
    plt.tight_layout()
    plt.savefig(DOCS / "fig4_scatter.png", dpi=150)
    plt.show()

    # Figure 5: Actual vs predicted temperature for bothh models
    test_dates = final_df["Date"].values[-len(y_test_sk):]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(test_dates, y_test_sk.values, label="Actual")
    plt.plot(test_dates, y_pred_lr, label="Predicted", linestyle="--")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Temperature Anomaly")
    plt.legend()
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.plot(test_dates, y_test_sk.values, label="Actual")
    plt.plot(test_dates, y_pred_rf, label="Predicted", linestyle="--")
    plt.title("Random Forest: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Temperature Anomaly")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(DOCS / "fig5_actual_vs_predicted.png", dpi=150)
    plt.show()

    # Figure 6: Human-driven vs natural driver importance-- just Random Forest
    human_features = {
        "CO2", "CH4", "N2O", "PrimaryEnergy", "LandUseCO2",
        "CementCO2", "FlaringCO2", "CO2_GrowthRate", "CH4_GrowthRate",
        "N2O_GrowthRate", "CO2_MA12", "CH4_MA12", "N2O_MA12", "CO2_Accel"
    }
    natural_features = {"TimeSinceBaseline", "MonthSin", "MonthCos", "Temp_Lag1", "Temp_Lag12"}

    rf_imp_full = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })

    rf_imp_full["Category"] = rf_imp_full["Feature"].apply(
        lambda f: "Human-Driven" if f in human_features
        else ("Natural/Temporal" if f in natural_features else "Other")
    )

    category_summary = rf_imp_full.groupby("Category")["Importance"].sum().reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(category_summary["Category"], category_summary["Importance"])
    plt.title("Human-Driven vs Natural/Temporal Drivers (Random Forest)")
    plt.xlabel("Driver Category")
    plt.ylabel("Cumulative Importance")
    plt.tight_layout()
    plt.savefig(DOCS / "fig6_human_vs_natural.png", dpi=150)
    plt.show()

    # Figure 7: Correlation with temperature
    num_df = final_df.drop(columns=["Date"]).select_dtypes(include=np.number)
    corr = num_df.corr()[["Temp"]].drop("Temp").sort_values("Temp", ascending=False)

    plt.figure(figsize=(6, 8))
    plt.barh(corr.index, corr["Temp"])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Pearson Correlation with Temperature Anomaly")
    plt.xlabel("Pearson r")
    plt.tight_layout()
    plt.savefig(DOCS / "fig7_correlation.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()