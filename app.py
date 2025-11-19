import io
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Genel ayarlar
# ------------------------------------------------------------
st.set_page_config(
    page_title="BG-EAF Arc Optimizer",
    layout="wide",
)


# ------------------------------------------------------------
# Yardımcı fonksiyonlar
# ------------------------------------------------------------
@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def add_panel_cooling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Panel soğutma kolonları varsa panel_Q_kW feature'ını ekler.
    Beklenen kolonlar:
        panel_T_in_C, panel_T_out_C, panel_flow_kg_s
    """
    required = ["panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s"]
    if all(c in df.columns for c in required):
        if "panel_Q_kW" not in df.columns:
            df = df.copy()
            cp_kJ = 4.18  # su için yaklaşık özgül ısı (kJ/kgK)
            dT = df["panel_T_out_C"] - df["panel_T_in_C"]
            # flow (kg/s) * cp (kJ/kgK) * dT (K) ~ kW
            df["panel_Q_kW"] = df["panel_flow_kg_s"] * cp_kJ * dT
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y, feature_cols


def train_rf_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": mse,
        "RMSE": rmse,
        "R2": float(r2_score(y_test, y_pred)),
    }

    return {
        "model": model,
        "metrics": metrics,
    }


def ensure_session_keys():
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None
    if "feature_cols" not in st.session_state:
        st.session_state["feature_cols"] = None
    if "target_col" not in st.session_state:
        st.session_state["target_col"] = None


def set_model(model, feature_cols: List[str], target_col: str, metrics: Dict):
    st.session_state["trained_model"] = model
    st.session_state["feature_cols"] = feature_cols
    st.session_state["target_col"] = target_col
    st.session_state["model_metrics"] = metrics


def get_model():
    return (
        st.session_state.get("trained_model"),
        st.session_state.get("feature_cols"),
        st.session_state.get("target_col"),
        st.session_state.get("model_metrics"),
    )


# ---------- CFD DEMO FONKSİYONU ---------------------------------
@st.cache_data
def generate_cfd_fields(
    power_kWh: float,
    oxygen_Nm3: float,
    tap_time_min: float,
    slag_foam_level: float = 0.5,
    nx: int = 80,
    ny: int = 80,
):
    """
    Gerçek CFD değil; EAF içi sıcaklık + hız alanına benzeyen sentetik bir alan üretir.
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(0.0, 2.0, ny)
    X, Y = np.meshgrid(x, y)

    # Arkın merkezi ve yarıçapı
    r = np.sqrt(X**2 + (Y - 1.4) ** 2)

    # Güce göre temel sıcaklık seviyesi
    base_T = 1580 + (power_kWh - 4000) * 0.03 + (oxygen_Nm3 - 180) * 0.4
    core_T = base_T + 80

    # Çekirdek: merkezde sıcaklık yüksek, r ile azalan Gauss
    T = core_T * np.exp(-(r / 0.45) ** 2) + base_T - 60

    # Tap süresi uzadıkça homojenleşme (gradient azalıyor)
    hom_factor = np.clip((tap_time_min - 45) / 20, 0.0, 1.0)
    T = base_T + (T - base_T) * (1.0 - 0.4 * hom_factor)

    # Slag köpüğü, üst bölgede ısı kaybını azaltır
    slag_height = 1.3 + slag_foam_level * 0.3
    foam_mask = Y >= slag_height
    T = np.where(foam_mask, T + 20 * slag_foam_level, T)

    # Basit swirl akış (sanki manyetik karıştırma varmış gibi)
    U = -(Y - 1.0)
    V = X
    vel_mag = np.sqrt(U**2 + V**2)

    # Oksijen arttıkça türbülans (swirl) bir miktar artıyor
    vel_mag *= 0.6 + (oxygen_Nm3 - 170) * 0.01

    return X, Y, T, vel_mag, slag_height


def ensure_cfd_defaults():
    if "cfd_power" not in st.session_state:
        st.session_state["cfd_power"] = 4200.0
    if "cfd_oxygen" not in st.session_state:
        st.session_state["cfd_oxygen"] = 190.0
    if "cfd_tap_time" not in st.session_state:
        st.session_state["cfd_tap_time"] = 52.0
    if "cfd_slag_foam" not in st.session_state:
        st.session_state["cfd_slag_foam"] = 0.5


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
ensure_session_keys()

st.sidebar.title("BG-EAF Arc Optimizer")

page = st.sidebar.radio(
    "Menü",
    (
        "1) Veri Yükleme & Keşif",
        "2) Model Eğitimi",
        "3) Tek Heat Tahmini",
        "4) Batch Tahmin (CSV)",
        "5) CFD Görselleştirme (Demo)",
    ),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "EAF / pota ocağı verisi üzerinden enerji / süre / sıcaklık tahmini, "
    "panel soğutma feature'ları ve demo CFD görselleştirmesi."
)


# ------------------------------------------------------------
# 1) Veri Yükleme & Keşif
# ------------------------------------------------------------
if page.startswith("1"):
    st.title("1) Veri Yükleme & Keşif")

    uploaded = st.file_uploader(
        "EAF / pota ocağı veri dosyanı (CSV) yükle",
        type=["csv"],
    )

    if uploaded is not None:
        df = load_csv(uploaded.getvalue())
        # Panel soğutma feature'ını otomatik ekle
        df = add_panel_cooling_features(df)
        st.session_state["raw_df"] = df

        st.success(
            f"**{uploaded.name}** yüklendi – satır: {df.shape[0]}, kolon: {df.shape[1]}"
        )

        tab1, tab2, tab3 = st.tabs(["Örnek Satırlar", "Kolon Bilgisi", "İstatistikler"])

        with tab1:
            st.subheader("İlk 50 satır")
            st.dataframe(df.head(50))

        with tab2:
            st.subheader("Kolonlar")
            info_df = pd.DataFrame(
                {
                    "Kolon": df.columns,
                    "Tip": [str(t) for t in df.dtypes],
                    "Boş Sayısı": df.isna().sum().values,
                }
            )
            st.dataframe(info_df)

        with tab3:
            st.subheader("Tanımlayıcı İstatistikler")
            st.dataframe(df.describe(include="all").transpose())

        st.info(
            "Sonraki adımda bir **hedef kolon** seçerek (ör. `tap_time_min`, "
            "`tap_temperature_C`, `melt_temperature_C`) model eğiteceğiz.\n\n"
            "Panel soğutma kolonları mevcutsa, `panel_Q_kW` feature'ı otomatik hesaplanır."
        )
    else:
        if st.session_state["raw_df"] is None:
            st.warning("Devam etmek için lütfen bir CSV dosyası yükle.")
        else:
            st.info("Mevcut veri seti session'da yüklü. Diğer menülerden devam edebilirsin.")


# ------------------------------------------------------------
# 2) Model Eğitimi
# ------------------------------------------------------------
elif page.startswith("2"):
    st.title("2) Model Eğitimi")

    df = st.session_state.get("raw_df")

    if df is None:
        st.warning("Önce **1) Veri Yükleme & Keşif** sekmesinden CSV yüklemen gerekiyor.")
    else:
        # Panel soğutma feature'ı yoksa burada da eklemeyi dene
        df = add_panel_cooling_features(df)
        st.session_state["raw_df"] = df

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if not numeric_cols:
            st.error(
                "Sayısal kolon bulunamadı. Model için en az bir sayısal kolon gerekli."
            )
        else:
            st.write("Veri boyutu:", df.shape)
            st.write("Sayısal kolonlar:", numeric_cols)

            target_col = st.selectbox(
                "Tahmin etmek istediğin hedef kolon (target):",
                options=numeric_cols,
            )

            default_features = [c for c in numeric_cols if c != target_col]
            feature_cols = st.multiselect(
                "Girdi (feature) kolonlar (hurda, enerji, sıcaklık, panel_Q_kW vb.):",
                options=numeric_cols,
                default=default_features,
            )

            test_size = st.slider(
                "Test oranı (train/test split)",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
            )

            n_estimators = st.slider(
                "Random Forest ağaç sayısı",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
            )

            if not feature_cols:
                st.error("En az bir feature kolon seçmelisin.")
            else:
                if st.button("Modeli Eğit"):
                    with st.spinner("Model eğitiliyor..."):
                        model_df = df[feature_cols + [target_col]].dropna()
                        X, y, _ = split_features_target(model_df, target_col)

                        result = train_rf_model(
                            X,
                            y,
                            test_size=test_size,
                            n_estimators=n_estimators,
                        )
                        model = result["model"]
                        metrics = result["metrics"]

                        set_model(model, feature_cols, target_col, metrics)

                    st.success("Model başarıyla eğitildi.")
                    st.subheader("Performans Metri̇kleri")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MAE", f"{metrics['MAE']:.3f}")
                    col2.metric("RMSE", f"{metrics['RMSE']:.3f}")
                    col3.metric("R2", f"{metrics['R2']:.3f}")
                    col4.metric("MSE", f"{metrics['MSE']:.3f}")

                    st.caption(
                        "Not: Bu metrikler seçilen veri alt kümesi, feature set ve "
                        "train/test oranına göre hesaplanmıştır."
                    )

            model, fcols, tcol, m = get_model()
            if model is not None:
                st.markdown("---")
                st.subheader("Mevcut Model Özeti")
                st.write("Hedef kolon:", f"`{tcol}`")
                st.write("Feature kolonlar:", fcols)
                if m:
                    st.write("Son eğitim metrikleri:", m)


# ------------------------------------------------------------
# 3) Tek Heat Tahmini
# ------------------------------------------------------------
elif page.startswith("3"):
    st.title("3) Tek Heat Tahmini")

    model, feature_cols, target_col, metrics = get_model()

    if model is None or feature_cols is None or target_col is None:
        st.warning(
            "Önce **2) Model Eğitimi** sekmesinden bir model eğitmen gerekiyor."
        )
    else:
        st.info(
            f"Mevcut model, **{target_col}** değerini tahmin ediyor. "
            "Aşağıya yeni bir heat için proses değerlerini girerek "
            "tahmini görebilirsin."
        )

        cols_per_row = 3
        input_values = []

        for i, fcol in enumerate(feature_cols):
            if i % cols_per_row == 0:
                row = st.columns(cols_per_row)
            with row[i % cols_per_row]:
                val = st.number_input(
                    fcol,
                    value=0.0,
                    step=1.0,
                    format="%.3f",
                )
                input_values.append(val)

        if st.button("Tahmini Hesapla"):
            arr = np.array(input_values, dtype=float).reshape(1, -1)
            pred = float(model.predict(arr)[0])
            st.success(f"Tahmin edilen **{target_col}**: `{pred:.3f}`")

            if metrics:
                st.caption(
                    f"Bu tahmin, R2={metrics['R2']:.3f} seviyesinde doğruluğa sahip "
                    "bir Random Forest modeli ile üretilmiştir."
                )


# ------------------------------------------------------------
# 4) Batch Tahmin (CSV)
# ------------------------------------------------------------
elif page.startswith("4"):
    st.title("4) Batch Tahmin (CSV)")

    model, feature_cols, target_col, metrics = get_model()

    if model is None or feature_cols is None or target_col is None:
        st.warning(
            "Önce **2) Model Eğitimi** sekmesinden bir model eğitmen gerekiyor."
        )
    else:
        st.info(
            "Feature kolonları içeren bir CSV yükleyerek toplu tahmin alabilirsin. "
            f"Beklenen kolonlar: {feature_cols}"
        )

        uploaded = st.file_uploader("Batch tahmin CSV yükle", type=["csv"])

        if uploaded is not None:
            batch_df = load_csv(uploaded.getvalue())
            batch_df = add_panel_cooling_features(batch_df)

            missing_cols = [c for c in feature_cols if c not in batch_df.columns]
            if missing_cols:
                missing_text = ", ".join(missing_cols)
                st.error(
                    f"Yüklenen dosyada aşağıdaki zorunlu kolonlar eksik: {missing_text}"
                )
            else:
                features = batch_df[feature_cols].values
                preds = model.predict(features)

                result_df = batch_df.copy()
                result_df[f"pred_{target_col}"] = preds

                st.subheader("Tahmin Sonuçları (ilk 50 satır)")
                st.dataframe(result_df.head(50))

                csv_buf = io.StringIO()
                result_df.to_csv(csv_buf, index=False)
                st.download_button(
                    "Sonuçları CSV olarak indir",
                    data=csv_buf.getvalue(),
                    file_name="arc_optimizer_predictions.csv",
                    mime="text/csv",
                )

                if metrics:
                    st.caption(
                        f"Tahminler, R2={metrics['R2']:.3f} seviyesinde doğruluğa sahip "
                        "bir Random Forest modeli ile üretilmiştir."
                    )


# ------------------------------------------------------------
# 5) CFD Görselleştirme (Demo)
# ------------------------------------------------------------
elif page.startswith("5"):
    st.title("5) CFD Görselleştirme (Demo)")

    ensure_cfd_defaults()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Proses Parametreleri")

        power_kWh = st.slider(
            "Güç (kWh)",
            min_value=3800.0,
            max_value=4600.0,
            value=float(st.session_state["cfd_power"]),
            step=10.0,
        )
        oxygen_Nm3 = st.slider(
            "Oksijen (Nm³)",
            min_value=150.0,
            max_value=230.0,
            value=float(st.session_state["cfd_oxygen"]),
            step=1.0,
        )
        tap_time_min = st.slider(
            "Tap Süresi (dk)",
            min_value=45.0,
            max_value=65.0,
            value=float(st.session_state["cfd_tap_time"]),
            step=0.5,
        )
        slag_foam_level = st.slider(
            "Slag Foam Seviyesi (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["cfd_slag_foam"]),
            step=0.1,
        )

        st.session_state["cfd_power"] = power_kWh
        st.session_state["cfd_oxygen"] = oxygen_Nm3
        st.session_state["cfd_tap_time"] = tap_time_min
        st.session_state["cfd_slag_foam"] = slag_foam_level

        st.markdown("---")
        st.caption(
            "Not: Bu sekme **gerçek CFD çözümü** değil, konsept göstermek için "
            "hazırlanmış sentetik bir sıcaklık + hız alanı görselleştirmesidir."
        )

    with col_right:
        X, Y, T, vel_mag, slag_h = generate_cfd_fields(
            power_kWh, oxygen_Nm3, tap_time_min, slag_foam_level
        )

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        t_plot = axs[0].contourf(X, Y, T, levels=30)
        axs[0].set_title("Sıcaklık Alanı (°C) – Kesit")
        axs[0].set_xlabel("Genişlik (normalize)")
        axs[0].set_ylabel("Yükseklik (normalize)")
        axs[0].axhline(slag_h, linestyle="--")
        fig.colorbar(t_plot, ax=axs[0], shrink=0.9)

        v_plot = axs[1].contourf(X, Y, vel_mag, levels=30)
        axs[1].set_title("Hız Büyüklüğü – Swirl / Karıştırma")
        axs[1].set_xlabel("Genişlik (normalize)")
        axs[1].set_ylabel("Yükseklik (normalize)")
        axs[1].axhline(slag_h, linestyle="--")
        fig.colorbar(v_plot, ax=axs[1], shrink=0.9)

        fig.tight_layout()
        st.pyplot(fig)
