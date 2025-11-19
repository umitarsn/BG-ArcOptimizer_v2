
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D için gerekli)
import streamlit as st


# ==========================
# Yardımcı 3D çizim fonksiyonları
# ==========================
def draw_box(ax, x, y, z, dx, dy, dz, color="gray", alpha=0.4):
    """Basit dikdörtgen prizma (bina / ekipman) çizer."""
    # Alt ve üst yüzler
    X = np.array([[x, x + dx], [x, x + dx]])
    Y = np.array([[y, y], [y + dy, y + dy]])
    Z_bottom = np.array([[z, z], [z, z]])
    Z_top = np.array([[z + dz, z + dz], [z + dz, z + dz]])

    ax.plot_surface(X, Y, Z_bottom, color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(X, Y, Z_top, color=color, alpha=alpha, linewidth=0)

    # Yan yüzler
    Y_side = np.array([[y, y], [y, y]])
    Z_side = np.array([[z, z + dz], [z, z + dz]])
    X_left = np.array([[x, x], [x + dx, x + dx]])
    X_right = np.array([[x + dx, x + dx], [x, x]])

    ax.plot_surface(X_left, Y_side, Z_side, color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(X_right, Y_side + dy, Z_side, color=color, alpha=alpha, linewidth=0)

    X_front = np.array([[x, x], [x, x]])
    Y_front = np.array([[y, y + dy], [y, y + dy]])
    ax.plot_surface(X_front, Y_front, Z_side, color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(X_front + dx, Y_front, Z_side, color=color, alpha=alpha, linewidth=0)


def draw_ferrochrome_plant(ax):
    """
    Çok basitleştirilmiş bir ferrochrome tesisi:
    - Ortada DC EAF
    - Yanda rectifier / trafo binası
    - Önde pota parkı + potalar
    - Solda hammadde / charge alanı
    """

    # Zemin
    Xg, Yg = np.meshgrid(np.linspace(-40, 40, 2), np.linspace(-20, 20, 2))
    Zg = np.zeros_like(Xg)
    ax.plot_surface(Xg, Yg, Zg, color="#444444", alpha=0.3)

    # DC EAF binası
    draw_box(ax, x=-5, y=-5, z=0, dx=10, dy=10, dz=8, color="#777777", alpha=0.5)

    # EAF gövdesi (silindir)
    r = 3.0
    z1, z2 = 0, 6
    theta = np.linspace(0, 2 * np.pi, 40)
    z_cyl = np.linspace(z1, z2, 20)
    T, Z_cyl = np.meshgrid(theta, z_cyl)
    X_cyl = r * np.cos(T)
    Y_cyl = r * np.sin(T)
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color="#999999", alpha=0.8)

    # Üst kapak
    ax.plot_trisurf(r * np.cos(theta), r * np.sin(theta),
                    np.full_like(theta, z2), color="#888888", alpha=0.8)

    # DC elektrot (tek)
    theta_e = np.linspace(0, 2 * np.pi, 20)
    r_e = 0.6
    z_e1, z_e2 = z2, z2 + 5
    Te, Ze = np.meshgrid(theta_e, np.linspace(z_e1, z_e2, 10))
    Xe = r_e * np.cos(Te)
    Ye = r_e * np.sin(Te)
    ax.plot_surface(Xe, Ye, Ze, color="#222222", alpha=1.0)

    # Rectifier / trafo binası (sağda)
    draw_box(ax, x=8, y=-4, z=0, dx=12, dy=8, dz=6, color="#5555aa", alpha=0.5)

    # Hammadde / charge alanı (solda)
    draw_box(ax, x=-25, y=-6, z=0, dx=12, dy=12, dz=5, color="#aa8844", alpha=0.5)

    # Pota park alanı (öne doğru)
    draw_box(ax, x=-4, y=8, z=0, dx=8, dy=8, dz=0.5, color="#666666", alpha=0.5)

    # 2 adet pota (silindirler)
    for offset in (-1.5, 1.5):
        r_p = 1.3
        z_p1, z_p2 = 0, 3
        th = np.linspace(0, 2 * np.pi, 30)
        z_p = np.linspace(z_p1, z_p2, 10)
        Th, Zp = np.meshgrid(th, z_p)
        Xp = - offset + r_p * np.cos(Th)
        Yp = 10 + r_p * np.sin(Th)
        ax.plot_surface(Xp, Yp, Zp, color="#bb7744", alpha=0.8)

    ax.set_title("Ferrochrome Tesisi – 3D Şema (DC EAF, Rectifier, Potlar)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


def draw_eaf_temperature(ax):
    """
    DC ark ocağının içinin 3D pseudo-CFD sıcaklık alanı.
    Karıştırma yok; eksen-simetrik Gauss tipi ark çekirdeği + banyo.
    """
    # R (0..Rmax), Z (taban..yükseklik)
    R_max = 3.0
    Z_min, Z_max = 0.0, 6.0
    nr, nz = 40, 40

    r = np.linspace(0, R_max, nr)
    z = np.linspace(Z_min, Z_max, nz)
    R, Z = np.meshgrid(r, z)

    # Basit sıcaklık modeli
    z_arc_center = 4.5
    r_arc_center = 0.0
    arc_radius = 1.0

    base_T = 1600.0
    core_T = 2800.0

    dist2 = (R - r_arc_center) ** 2 + (Z - z_arc_center) ** 2
    T = base_T + (core_T - base_T) * np.exp(-dist2 / (2 * arc_radius ** 2))

    # Banyo çekirdeği ve üst gaz bölgesi ayarı
    T += 80.0 * np.exp(-(Z - 1.5) ** 2 / 1.5)          # banyo çekirdeği
    T -= 100.0 * np.clip((Z - 5.0) / 2.0, 0, 1)        # üst bölge soğuma

    # 3D dönüştürme (eksen simetrik)
    theta = np.linspace(0, 2 * np.pi, 60)
    Theta, _ = np.meshgrid(theta, z)

    R3 = np.repeat(R[:, :, np.newaxis], len(theta), axis=2)
    Z3 = np.repeat(Z[:, :, np.newaxis], len(theta), axis=2)
    T3 = np.repeat(T[:, :, np.newaxis], len(theta), axis=2)
    Theta3 = np.repeat(Theta[np.newaxis, :, :], nr, axis=0)

    X = R3 * np.cos(Theta3)
    Y = R3 * np.sin(Theta3)

    norm = plt.Normalize(T3.min(), T3.max())
    colors = cm.inferno(norm(T3))

    # Mesh yoğun → z yönünde seyrelt
    step_z = 2
    nz = Z3.shape[0]
    for k in range(0, nz, step_z):
        ax.plot_surface(
            X[k, :, :],
            Y[k, :, :],
            Z3[k, :, :],
            facecolors=colors[k, :, :],
            rstride=1,
            cstride=2,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    mappable = cm.ScalarMappable(cmap=cm.inferno, norm=norm)
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="T (°C)")

    ax.set_title("DC Ark Ocağı – 3D Pseudo CFD Sıcaklık (Karıştırma Yok)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


# ==========================
# Streamlit Arayüzü
# ==========================
st.set_page_config(page_title="Ferrochrome 3D Layout & DC EAF CFD", layout="wide")

st.title("Ferrochrome Tesisi 3D Görselleştirme + DC Ark Ocağı İç Sıcaklık (Demo)")

with st.sidebar:
    st.header("Görünüm Ayarları")

    elev1 = st.slider("Tesis görünümü Elevation", 5, 60, 20, 1)
    azim1 = st.slider("Tesis görünümü Azimuth", -120, 60, -60, 1)

    st.markdown("---")

    elev2 = st.slider("EAF içi sıcaklık Elevation", 5, 60, 25, 1)
    azim2 = st.slider("EAF içi sıcaklık Azimuth", -60, 120, 35, 1)

    st.markdown("---")
    st.caption(
        "Bu sayfa gerçek CFD çözücüsü kullanmaz; konsept anlatmak için "
        "hazırlanmış basitleştirilmiş 3D alan gösterimidir."
    )

# Figure
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
draw_ferrochrome_plant(ax1)
ax1.view_init(elev=elev1, azim=azim1)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
draw_eaf_temperature(ax2)
ax2.view_init(elev=elev2, azim=azim2)

plt.tight_layout()
st.pyplot(fig)
