import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------------------------------------------------------------
# 1. SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="BG-ArcOptimizer v2",
    layout="wide",
    page_icon="⚡"
)

# ------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------

def create_gauge_chart(value, target=1620):
    """Sıcaklık için ibreli gösterge (Gauge) oluşturur."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tahmini Döküm Sıcaklığı (°C)", 'font': {'size': 20}},
        delta = {'reference': target, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [1500, 1750], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1500, 1600], 'color': '#4dabf5'},  # Soğuk (Mavi)
                {'range': [1600, 1640], 'color': '#66ff66'},  # İdeal (Yeşil)
                {'range': [1640, 1750], 'color': '#ff6666'}], # Sıcak (Kırmızı)
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1700}}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def generate_cfd_fields(power, oxygen, time, foam_level):
    """Basit CFD simülasyonu (Görsel amaçlı sahte veri)"""
    nx, ny = 50, 50
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    
    # Sıcaklık dağılımı (Merkezde ark var, kenarlar soğuk)
    center_x, center_y = 5.0, 5.0
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    # Güç arttıkça merkez ısınır
    base_temp = 1500 + (power / 100)
    temp_field = base_temp * np.exp(-dist_sq / 10.0)
    
    return X, Y, temp_field

# ------------------------------------------------------------
# 3. ANA UYGULAMA AKIŞI
# ------------------------------------------------------------
def main():
    st.title("⚡ Elektrik Ark Ocağı - Akıllı Karar Destek Sistemi")
    
    # --- VERİ YÜKLEME SEÇENEĞİ (DEMO vs GERÇEK) ---
    st.sidebar.header("📂 Veri Kaynağı")
    data_mode = st.sidebar.radio(
        "Çalışma Modu Seçiniz:",
        ("Demo Verileri (Otomatik)", "Kendi Dosyamı Yükle (CSV)")
    )

    df = None
    
    if data_mode == "Demo Verileri (Otomatik)":
        try:
            # Demo dosyası yolunu kontrol et
            df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv")
            st.info(f"ℹ️ **Demo Modu:** {len(df)} satırlık simülasyon verisi kullanılıyor.")
        except FileNotFoundError:
            st.error("⚠️ Demo veri dosyası bulunamadı! Lütfen önce veri üretim kodunu (generate_data.py) çalıştırın.")
            st.stop()
    else:
        uploaded_file = st.sidebar.file_uploader("CSV Dosyanızı Sürükleyin", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dosya Yüklendi: {len(df)} satır.")
        else:
            st.warning("👈 Lütfen sol menüden bir CSV dosyası yükleyin veya Demo moduna geçin.")
            st.stop()

    # --- MODEL EĞİTİMİ ---
    target_col = "tap_temperature_C"
    
    # CSV içinde hedef kolon var mı kontrol et
    if target_col not in df.columns:
        st.error(f"Hata: CSV dosyasında '{target_col}' sütunu bulunamadı.")
        st.stop()

    # Gereksiz kolonları çıkar
    drop_cols = ["heat_id", "tap_time_min", "melt_temperature_C", "panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Başarım Metrikleri
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Sekmeler (Tabs)
    tab_main, tab_cfd = st.tabs(["📊 Karar Destek Paneli", "🔥 CFD Simülasyonu (Demo)"])

    # --- TAB 1: KARAR DESTEK & MALİYET ---
    with tab_main:
        with st.expander("📈 Model Doğruluk Oranlarını Göster"):
            c1, c2 = st.columns(2)
            c1.metric("Hata Payı (MAE)", f"±{mae:.1f} °C")
            c2.metric("Model Güveni (R²)", f"%{r2*100:.1f}")

        st.markdown("---")

        # Kullanıcı Girdileri (Simülasyon)
        st.sidebar.markdown("---")
        st.sidebar.header("🎛️ Simülasyon Parametreleri")
        
        input_data = {}
        for col in X.columns:
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            mean_v = float(df[col].mean())
            input_data[col] = st.sidebar.slider(f"{col}", min_v, max_v, mean_v)
        
        # Maliyet Girdileri
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Birim Fiyatlar ($)")
        price_elec = st.sidebar.number_input("Elektrik ($/kWh)", 0.05, 0.50, 0.10)
        price_oxy = st.sidebar.number_input("Oksijen ($/Nm3)", 0.05, 1.00, 0.15)
        price_electrode = st.sidebar.number_input("Elektrot ($/kg)", 1.0, 10.0, 4.5)
        electrode_rate = st.sidebar.number_input("Elektrot Sarfiyatı (kg/ton)", 1.0, 5.0, 1.8)

        # Tahmin Yap
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        # 1. Üst Kısım: Gösterge ve Tavsiye
        col_gauge, col_advice = st.columns([2, 2])
        
        with col_gauge:
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
        
        with col_advice:
            st.subheader("🤖 Operatör Asistanı")
            if prediction < 1600:
                st.error(f"⚠️ **Düşük Sıcaklık ({prediction:.1f}°C)**")
                st.write("👉 Döküm yapılamaz. Enerji girişini artırın veya hurda şarjını erteleyin.")
            elif 1600 <= prediction <= 1640:
                st.success(f"✅ **İdeal Döküm Aralığı ({prediction:.1f}°C)**")
                st.write("👉 Mevcut parametreler optimum seviyede. Müdahale gerekmez.")
            else:
                st.warning(f"🔥 **Gereksiz Aşırı Isınma ({prediction:.1f}°C)**")
                st.write("👉 Enerji israfı var. Gücü kesebilir veya oksijeni azaltabilirsiniz.")

        st.divider()

        # 2. Alt Kısım: Maliyet ve Açıklama
        col_cost, col_feat = st.columns(2)

        with col_cost:
            st.subheader("💵 Maliyet Analizi (Tahmini)")
            
            # Değerleri al
            pwr = input_data.get('power_kWh', 0)
            oxy = input_data.get('oxygen_Nm3', 0)
            
            cost_e = pwr * price_elec
            cost_o = oxy * price_oxy
            cost_el = 100 * electrode_rate * price_electrode # 100 ton varsayımı
            total = cost_e + cost_o + cost_el

            st.dataframe(pd.DataFrame({
                "Kalem": ["Elektrik", "Oksijen", "Elektrot", "TOPLAM"],
                "Maliyet ($)": [f"{cost_e:.2f}", f"{cost_o:.2f}", f"{cost_el:.2f}", f"{total:.2f}"]
            }), hide_index=True, use_container_width=True)
            
        with col_feat:
            st.subheader("🔍 Neden Bu Sonuç?")
            importances = pd.DataFrame({
                'Faktör': X.columns,
                'Etki': model.feature_importances_
            }).sort_values(by='Etki', ascending=False).head(5)
            
            st.bar_chart(importances.set_index('Faktör'))
            st.caption("Modelin sıcaklık tahmininde en çok dikkate aldığı 5 parametre.")

    # --- TAB 2: CFD GÖRÜNÜMÜ (ESKİ KODUNUZUN TEMİZLENMİŞ HALİ) ---
    with tab_cfd:
        st.subheader("Sanal CFD Isı Dağılımı")
        st.info("Bu ekran, parametrelerin fırın içindeki ısı dağılımını nasıl etkilediğini simüle eder.")
        
        pwr_cfd = input_data.get('power_kWh', 4000)
        oxy_cfd = input_data.get('oxygen_Nm3', 200)
        
        X_grid, Y_grid, T_field = generate_cfd_fields(pwr_cfd, oxy_cfd, 50, 0.5)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.contourf(X_grid, Y_grid, T_field, levels=20, cmap='inferno')
        fig.colorbar(c, label='Sıcaklık (°C)')
        ax.set_title(f"EAF Taban Sıcaklık Dağılımı (Güç: {pwr_cfd} kWh)")
        ax.set_xlabel("Fırın Genişliği (m)")
        ax.set_ylabel("Fırın Derinliği (m)")
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
