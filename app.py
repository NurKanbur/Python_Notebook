import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from catboost import CatBoostRegressor
from deap import base, creator, tools, algorithms
import plotly.express as px

# MODELLERİ YÜKLE
prd_model = joblib.load("random_forest_prd_model_New.pkl")
act_model = joblib.load("ACT_RanFo_new.pkl")
uretim_model = joblib.load("random_forest_uretim_model_New.pkl")

durus_model = CatBoostRegressor()
durus_model.load_model("catboost_model_durus_enc.cbm")

# ENCODER'LARI YÜKLE
makine_encoder = joblib.load("encoder_makine.pkl")
typecode_encoder = joblib.load("encoder_typecode.pkl")

# STREAMLIT ARAYÜZ
st.set_page_config(page_title="Üretim Optimizasyonu", layout="wide")
st.title("🔧 Üretim Optimizasyon Sistemi")
st.markdown("Genetik algoritma ile optimum üretim parametrelerini keşfedin.")

makine_list = makine_encoder.classes_.tolist()
typecode_list = typecode_encoder.classes_.tolist()

MakineNo_input = st.selectbox("🧵 Makine No seçiniz", options=makine_list)
TypeCode_input = st.selectbox("📦 TypeCode seçiniz", options=typecode_list)

if MakineNo_input and TypeCode_input:
    MakineNo = int(makine_encoder.transform([MakineNo_input])[0])
    TypeCode = int(typecode_encoder.transform([TypeCode_input])[0])

    # GENETİK ALGORİTMA PARAMETRELERİ
    POP_SIZE = 10
    NGEN = 5
    CXPB = 0.7
    MUTPB = 0.1

    def eval_multi_objective(ind):
        rpm, ds4_rpm, durus_suresi_sn = [float(i) for i in ind]
        try:
            durus_dk = durus_suresi_sn / 60
            atki = (1440 - durus_dk) * rpm

            act_input = [[MakineNo, TypeCode, ds4_rpm, rpm, durus_suresi_sn, 0, atki]]
            act_df = pd.DataFrame(act_input, columns=act_model.feature_names_in_)
            act_pred = float(act_model.predict(act_df)[0])

            prd_input = [[MakineNo, TypeCode, ds4_rpm, rpm, durus_suresi_sn, atki, act_pred]]
            prd_df = pd.DataFrame(prd_input, columns=prd_model.feature_names_in_)
            prd_pred = float(prd_model.predict(prd_df)[0])

            # Ceza: ACT > PRD ise büyük ceza ver
            if act_pred > prd_pred:
                return 10000, 0, 0, 0  # Duruşu çok yüksek yaparak cezalandırıyoruz

            durus_input = [[MakineNo, TypeCode, rpm, atki, act_pred, prd_pred]]
            durus_pred = float(durus_model.predict(durus_input)[0])

            uretim_input = [[MakineNo, TypeCode, ds4_rpm, rpm, durus_pred, atki, act_pred]]
            uretim_df = pd.DataFrame(uretim_input, columns=uretim_model.feature_names_in_)
            uretim_pred = float(uretim_model.predict(uretim_df)[0])

            return -durus_pred, act_pred, prd_pred, uretim_pred

        except Exception:
            return 10000, 0, 0, 0  # Hata durumunda da cezalandır

    # GA TANIMLARI
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, 1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_rpm", random.uniform, 400, 800)
    toolbox.register("attr_ds4rpm", random.uniform, 400, 600)
    toolbox.register("attr_durus", random.uniform, 100, 5000)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_rpm, toolbox.attr_ds4rpm, toolbox.attr_durus), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[400, 400, 100],
                     up=[900, 900, 3000],
                     eta=20.0, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", eval_multi_objective)

    if st.button("🚀 Genetik Algoritmayı Çalıştır"):
        with st.spinner("Optimizasyon çalışıyor..."):
            pop = toolbox.population(n=POP_SIZE)
            hof = tools.ParetoFront()
            algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE,
                                      cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                      halloffame=hof, verbose=False)

        all_results = []
        for ind in pop:
            durus, act, prd, uretim = eval_multi_objective(ind)
            rpm, ds4rpm, durus_sure_sn = ind
            durus_dk = durus_sure_sn / 60
            atki = (1440 - durus_dk) * rpm
            is_pareto = ind in hof

            all_results.append({
                "RPM": int(rpm),
                "Duruş Süresi(sn)": int(round(-durus)),
                "ATKI": int(atki),
                "Tahmin ACT": round(act, 2),
                "Tahmin PRD": round(prd, 2),
                "ÜRETİM METRE": round(uretim, 2),
                "Pareto": is_pareto
            })

           # ... (önceki kodlar değişmeden devam ediyor)

        df_all = pd.DataFrame(all_results)

        st.success("✅ Optimizasyon tamamlandı.")
        st.subheader("📊 Tüm Çözümler (Pareto-optimal işaretli)")
        st.dataframe(df_all.sort_values("Pareto", ascending=False), use_container_width=True)

        # === ORTALAMA KIYASLAMA ===
        ortalama_uretim = 409.840538
        ortalama_durus = 5120.269438

        pareto_df = df_all[df_all["Pareto"]]
        if not pareto_df.empty:
            en_iyi_cozum = pareto_df.sort_values("ÜRETİM METRE", ascending=False).iloc[0]

            uretim_yuzde_artis = ((en_iyi_cozum["ÜRETİM METRE"] - ortalama_uretim) / ortalama_uretim) * 100
            durus_yuzde_azalis = ((ortalama_durus - en_iyi_cozum["Duruş Süresi(sn)"]) / ortalama_durus) * 100

            st.markdown("### 📈 Verimlilik Karşılaştırması")
            st.markdown(f"""
            - 🔼 **Üretim Artışı:** {uretim_yuzde_artis:.2f} %
            - 🔽 **Duruş Azalışı:** {durus_yuzde_azalis:.2f} %
            """)
        else:
            st.warning("Pareto-optimal çözüm bulunamadı. Karşılaştırma yapılamadı.")

        # 🎯 Pareto sınır grafiği
        from scipy.spatial import ConvexHull
        import plotly.graph_objects as go

        pts = pareto_df[["ÜRETİM METRE", "Duruş Süresi(sn)"]].to_numpy()
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
        else:
            hull_pts = pts

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_all["ÜRETİM METRE"], y=df_all["Duruş Süresi(sn)"],
            mode="markers", name="Tüm Çözümler",
            marker=dict(color="lightgray", size=6)
        ))

        fig.add_trace(go.Scatter(
            x=pareto_df["ÜRETİM METRE"], y=pareto_df["Duruş Süresi(sn)"],
            mode="markers", name="Pareto Çözümler",
            marker=dict(color="red", size=8)
        ))

        if len(hull_pts) > 0:
            hull_loop = np.vstack([hull_pts, hull_pts[0]])
            fig.add_trace(go.Scatter(
                x=hull_loop[:, 0], y=hull_loop[:, 1],
                mode="lines", name="Pareto Sınırı",
                line=dict(color="blue", dash="dash")
            ))

        fig.update_layout(
            title="Üretim Metre vs Duruş Süresi (Pareto Sınırı)",
            xaxis_title="Üretim Metre",
            yaxis_title="Duruş Süresi (sn)"
        )
        st.plotly_chart(fig, use_container_width=True)
