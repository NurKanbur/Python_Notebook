import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import xgboost as xgb
import plotly.graph_objects as go

# Modelleri yükleme
with open("sinem_model_1.pkl", "rb") as f:
    sinem_prophet = pickle.load(f)
with open("deniz_model_2.pkl", "rb") as f:
    deniz_prophet = pickle.load(f)
with open("sinem_xgb_regressor.pkl", "rb") as f:
    sinem_xgb = pickle.load(f)
with open("deniz_xgb_regressor.pkl", "rb") as f:
    deniz_xgb = pickle.load(f)

# Kullanıcı girişleri
st.title("Jeotermal Enerji Tahmin Sistemi")
santral = st.selectbox("Santral Seçiniz:", ["Sinem Santrali", "Deniz Santrali"])
model = st.selectbox("Model Seçiniz:", ["Prophet", "XGBoost"])
gun_sayisi = st.number_input("Tahmin Edilecek Saat Sayısı", min_value=1, max_value=365, value=30)

# Veri yükleme
data = pd.read_excel("kullanilacak_veriseti.xlsx")
data["date"] = pd.to_datetime(data["date"])

# Tahmin süreci
if st.button("Tahmin Yap"):
    future_df = pd.DataFrame()
    future_df["date"] = pd.date_range(start="2024-08-12 09:00:25", periods=gun_sayisi, freq="H")
    
    if santral == "Sinem Santrali":
        hedef = "baskilanmis_degerler"
        if model == "Prophet":
            future_df = future_df.rename(columns={"date": "ds"})
            future_df["sinem_guc_bop"] = data["sinem_guc_bop"].iloc[:gun_sayisi].values
            future_df["sinem_guc_gross"] = data["sinem_guc_gross"].iloc[:gun_sayisi].values
            forecast = sinem_prophet.predict(future_df)
            gercek = data[["date", hedef]].iloc[:gun_sayisi].copy()
            gercek["tahmin"] = forecast["yhat"].values
        elif model == "XGBoost":
            data["timestamp"] = data["date"].astype('int64') // 10**9
            data.set_index("timestamp", inplace=True)
            features = ["sinem_guc_bop", "sinem_guc_gross"]
            forecast = sinem_xgb.predict(data[features].iloc[:gun_sayisi])
            gercek = data[["date", hedef]].iloc[:gun_sayisi].copy()
            gercek["tahmin"] = forecast[:gun_sayisi]
    else:
        hedef = "deniz_baskilanmis_degerler"
        if model == "Prophet":
            future_df["deniz_guc_gross"] = data["deniz_guc_gross"].iloc[:gun_sayisi].values
            future_df["deniz_guc_teias"] = data["deniz_guc_teias"].iloc[:gun_sayisi].values
            future_df["deniz_basinc_con"] = data["deniz_basinc_con"].iloc[:gun_sayisi].values
            future_df["deniz_debi_gm17"] = data["deniz_debi_gm17"].iloc[:gun_sayisi].values
            future_df["deniz_debi_cikis"] = data["deniz_debi_cikis"].iloc[:gun_sayisi].values
            future_df["deniz_farkbasinc_l1u"] = data["deniz_farkbasinc_l1u"].iloc[:gun_sayisi].values
            future_df["deniz_farkbasinc_l2u"] = data["deniz_farkbasinc_l2u"].iloc[:gun_sayisi].values
            future_df = future_df.rename(columns={"date": "ds"})
            forecast = deniz_prophet.predict(future_df)
            gercek = data[["date", hedef]].iloc[:gun_sayisi].copy()
            gercek["tahmin"] = forecast["yhat"].values
        elif model == "XGBoost":
            data["timestamp"] = data["date"].astype('int64') // 10**9
            data.set_index("timestamp", inplace=True)
            features = ["deniz_guc_gross", "deniz_guc_teias", "deniz_basinc_con", 
                        "deniz_debi_gm17", "deniz_debi_cikis", 
                        "deniz_farkbasinc_l1u", "deniz_farkbasinc_l2u"]
            forecast = deniz_xgb.predict(data[features].iloc[:gun_sayisi])
            gercek = data[["date", hedef]].iloc[:gun_sayisi].copy()
            gercek["tahmin"] = forecast[:gun_sayisi]

    # Plotly ile interaktif grafik
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=gercek["date"], y=gercek[hedef],
                             mode='lines+markers',
                             name='Gerçek Değerler',
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=gercek["date"], y=gercek["tahmin"],
                             mode='lines+markers',
                             name='Tahmin Değerleri',
                             line=dict(color='red')))

    fig.update_layout(
        title="Gerçek vs Tahmin Edilen Değerler",
        xaxis_title="Tarih",
        yaxis_title="Üretim",
        legend=dict(x=0, y=1),
        hovermode="x unified",
        template="plotly_white",
        autosize=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Hata hesaplamaları
    mae = mean_absolute_error(gercek[hedef], gercek["tahmin"])
    rmse = np.sqrt(mean_squared_error(gercek[hedef], gercek["tahmin"]))

    st.write(f"Model Güven Yüzdesi (%): {100 - (mae / gercek[hedef].mean()) * 100:.2f}")
    st.write(f"Hata Payı Oranı: {mae / gercek[hedef].mean():.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    # Gerçek ve tahmin edilen değerlerin gösterimi
    st.write("Gerçek ve Tahmin Edilen Değerler:")
    st.dataframe(gercek[["date", hedef, "tahmin"]])
