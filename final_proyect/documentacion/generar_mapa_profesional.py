import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import LightSource, ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Rectangle

# Base paths relativos y autónomos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPAS_DIR = os.path.join(SCRIPT_DIR, "mapas_qgis")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Cargar rasters de probabilidad y AOA
with rasterio.open(os.path.join(MAPAS_DIR, "prob_logreg.tif")) as s:
    map_lr = s.read(1)
    b = s.bounds
    transform = s.transform
    crs = s.crs
    H, W = s.shape

with rasterio.open(os.path.join(MAPAS_DIR, "prob_rf.tif")) as s:
    map_rf = s.read(1)

with rasterio.open(os.path.join(MAPAS_DIR, "prob_mlp.tif")) as s:
    map_mlp = s.read(1)

with rasterio.open(os.path.join(MAPAS_DIR, "aoa_mask.tif")) as s:
    raw_aoa = s.read(1)
    map_aoa = np.full_like(map_lr, np.nan, dtype=np.float32)
    map_aoa[raw_aoa == 1] = 1.0 # Dentro
    map_aoa[raw_aoa == 2] = 0.0 # Fuera

# 2. Cargar DEM de 120 m (autocontenido en mapas_qgis)
dem_120m_path = os.path.join(MAPAS_DIR, "dem_120m.tif")
if os.path.exists(dem_120m_path):
    with rasterio.open(dem_120m_path) as s_dem:
        dem_120m = s_dem.read(1)
else:
    # Fallback si se ejecuta cerca de fuentes originales
    raw_candidates = [
        os.path.join(SCRIPT_DIR, "..", r"LC09_L2SP_038038_20260925_20260926_02_T1\elevaciones30.tif"),
        os.path.join(SCRIPT_DIR, "elevaciones30.tif")
    ]
    raw_dem = next((p for p in raw_candidates if os.path.exists(p)), None)
    if raw_dem:
        dem_120m = np.empty((H, W), dtype=np.float32)
        with rasterio.open(raw_dem) as s_dem:
            reproject(
                source=rasterio.band(s_dem, 1),
                destination=dem_120m,
                src_transform=s_dem.transform,
                src_crs=s_dem.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear
            )
    else:
        raise FileNotFoundError("No se localizó dem_120m.tif en mapas_qgis.")

valid_mask = ~np.isnan(map_lr)
dem_clean = np.where(dem_120m < -100, 0.0, dem_120m)
dem_clean = np.where(np.isnan(dem_clean), 0.0, dem_clean)

# Calcular sombreado de relieve (hillshade) de alto contraste
ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(dem_clean, dx=120, dy=120, vert_exag=4.5)
hs_norm = (hs - hs.min()) / (hs.max() - hs.min())
hs_rgb = np.dstack([hs_norm, hs_norm, hs_norm])

# Extensión territorial en kilómetros
extent_km = [b.left / 1000.0, b.right / 1000.0, b.bottom / 1000.0, b.top / 1000.0]

# 3. Mezcla RGBA donde el relieve realza la estructura geológica
cmap_prob = plt.cm.viridis
cmap_aoa = ListedColormap(['#d95f02', '#1b9e77'])

def create_prob_rgba(data):
    norm_d = np.nan_to_num(data, nan=0.0)
    c_prob = cmap_prob(norm_d)[:, :, :3]
    blended = np.clip(0.60 * c_prob + 0.40 * hs_rgb, 0.0, 1.0)
    return np.dstack([blended, valid_mask.astype(np.float32)])

def create_aoa_rgba(data):
    norm_d = np.nan_to_num(data, nan=0.0)
    c_aoa = cmap_aoa(norm_d)[:, :, :3]
    blended = np.clip(c_aoa * (0.38 + 0.82 * hs_norm[:, :, np.newaxis]), 0.0, 1.0)
    return np.dstack([blended, valid_mask.astype(np.float32)])

rgba_lr = create_prob_rgba(map_lr)
rgba_rf = create_prob_rgba(map_rf)
rgba_mlp = create_prob_rgba(map_mlp)
rgba_aoa = create_aoa_rgba(map_aoa)

# 4. Coordenadas de sitios de interés en km (UTM 11N)
cp_x, cp_y = 662.8, 3586.1 # Centroide CFE Cerro Prieto
cp_poly_x = [658.9, 665.8, 665.8, 658.9, 658.9]
cp_poly_y = [3582.3, 3582.3, 3589.5, 3589.5, 3582.3]

ch_x, ch_y = 685.295, 3481.121 # El Chinero
sf_x, sf_y = 706.176, 3434.376 # San Felipe

y_paralelo_32 = 3542.2 # Paralelo 32 N

# 5. Configuración de Figura IEEE (300 DPI)
fig, axes = plt.subplots(1, 4, figsize=(11.8, 6.2), dpi=300)
fig.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.24, wspace=0.08)

panels_data = [
    (axes[0], rgba_lr, "(a) Regresión Logística"),
    (axes[1], rgba_rf, "(b) Random Forest"),
    (axes[2], rgba_mlp, "(c) MLP Ensamble"),
    (axes[3], rgba_aoa, "(d) Área de Aplicabilidad")
]

halo = [pe.withStroke(linewidth=2.5, foreground='white')]
halo_dark = [pe.withStroke(linewidth=2.5, foreground='black')]
arrow_halo = [pe.withStroke(linewidth=2.5, foreground='white')]

def draw_north_arrow(ax):
    nx = extent_km[0] + 0.90 * (extent_km[1] - extent_km[0])
    ny = extent_km[2] + 0.925 * (extent_km[3] - extent_km[2])
    w = 0.028 * (extent_km[1] - extent_km[0])
    h = 0.046 * (extent_km[3] - extent_km[2])
    
    left_tri = Polygon([[nx, ny + h], [nx - w, ny], [nx, ny + h*0.25]], closed=True, 
                       facecolor='black', edgecolor='black', linewidth=0.5, zorder=15)
    right_tri = Polygon([[nx, ny + h], [nx + w, ny], [nx, ny + h*0.25]], closed=True, 
                        facecolor='white', edgecolor='black', linewidth=0.5, zorder=15)
    ax.add_patch(left_tri)
    ax.add_patch(right_tri)
    ax.text(nx, ny + h + 1.8, 'N', ha='center', va='bottom', fontsize=8, fontweight='bold', 
            color='black', path_effects=halo, zorder=16)

# Renderizado de paneles
for ax, rgba, title in panels_data:
    im = ax.imshow(rgba, extent=extent_km, origin='upper', zorder=2)
    
    # Línea discontinua del Paralelo 32 N (costura cartográfica)
    ax.axhline(y_paralelo_32, color='white', linestyle='--', linewidth=0.8, alpha=0.90, zorder=4)
    ax.text(extent_km[0] + 3.0, y_paralelo_32 + 1.5, "32° N", color='white', fontsize=6.5, 
            fontweight='bold', path_effects=halo_dark, zorder=5)
    
    # Sitios de Referencia
    # 1. CFE Cerro Prieto (Polígono delimitador rojo)
    ax.plot(cp_poly_x, cp_poly_y, color='#b30000', linewidth=1.3, zorder=6)
    ax.text(cp_x + 4.0, cp_y + 3.0, "CFE Cerro Prieto", fontsize=6.5, fontweight='bold', 
            color='#800000', path_effects=halo, zorder=8)
    
    # 2. El Chinero: Flecha indicadora magenta + etiqueta
    ax.annotate(
        "El Chinero",
        xy=(ch_x, ch_y),
        xytext=(ch_x + 14.0, ch_y - 1.0),
        fontsize=6.8,
        fontweight='bold',
        color='#4d004b',
        path_effects=halo,
        arrowprops=dict(
            arrowstyle="->,head_width=0.28,head_length=0.38",
            color='#e7298a',
            lw=1.3,
            shrinkA=1,
            shrinkB=4,
            path_effects=arrow_halo
        ),
        zorder=12
    )
    
    # 3. San Felipe (Punto azul + etiqueta)
    ax.scatter(sf_x, sf_y, color='#1f78b4', marker='o', s=24, edgecolors='black', linewidth=0.7, zorder=7)
    ax.text(sf_x - 3.5, sf_y - 2.0, "San Felipe", fontsize=6.5, fontweight='bold', 
            ha='right', color='#08306b', path_effects=halo, zorder=8)
    
    # Flecha de Norte
    draw_north_arrow(ax)
    
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=6)
    ax.set_xlabel("Este (km)", fontsize=7.5, fontweight='bold')
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.4)
    ax.set_xlim(extent_km[0], extent_km[1])
    ax.set_ylim(extent_km[2], extent_km[3])

axes[0].set_ylabel("Norte UTM 11N (km)", fontsize=7.5, fontweight='bold')
for ax in axes[1:]:
    ax.set_yticklabels([])

# Escala gráfica externa sobre la barra de color
scale_ax = fig.add_axes([0.24, 0.135, 0.26, 0.045])
scale_ax.set_xlim(0, 50)
scale_ax.set_ylim(0, 10)
scale_ax.axis('off')

bar_h = 3.5
scale_ax.add_patch(Rectangle((0, 0), 25, bar_h, facecolor='black', edgecolor='black', linewidth=0.7))
scale_ax.add_patch(Rectangle((25, 0), 25, bar_h, facecolor='white', edgecolor='black', linewidth=0.7))

scale_ax.text(0, bar_h + 1.2, "0", ha='center', va='bottom', fontsize=6.5, fontweight='bold')
scale_ax.text(25, bar_h + 1.2, "25", ha='center', va='bottom', fontsize=6.5, fontweight='bold')
scale_ax.text(50, bar_h + 1.2, "50 km", ha='center', va='bottom', fontsize=6.5, fontweight='bold')
scale_ax.text(25, -2.0, "Escala Gráfica", ha='center', va='top', fontsize=6.5, fontweight='bold', color='#333333')

# Barras de color inferiores
cbar_ax1 = fig.add_axes([0.08, 0.065, 0.58, 0.030])
sm_prob = plt.cm.ScalarMappable(cmap=cmap_prob, norm=plt.Normalize(vmin=0, vmax=1))
sm_prob.set_array([])
cb1 = fig.colorbar(sm_prob, cax=cbar_ax1, orientation='horizontal')
cb1.set_label("Puntuación del modelo P(x) [Afinidad espacial relativa]", fontsize=7.5, fontweight='bold')
cb1.ax.tick_params(labelsize=6.5)

cbar_ax2 = fig.add_axes([0.76, 0.065, 0.18, 0.030])
sm_aoa = plt.cm.ScalarMappable(cmap=cmap_aoa, norm=plt.Normalize(vmin=0, vmax=1))
sm_aoa.set_array([])
cb2 = fig.colorbar(sm_aoa, cax=cbar_ax2, orientation='horizontal', ticks=[0.25, 0.75])
cb2.ax.set_xticklabels(['Fuera', 'Dentro (62.8%)'], fontsize=6.5, fontweight='bold')
cb2.set_label("AOA (DI <= 2.13)", fontsize=7.5, fontweight='bold')

out_paths = [
    os.path.join(FIGURES_DIR, "mapa_regional_profesional.png"),
    os.path.join(FIGURES_DIR, "grafico_10_mapa_susceptibilidad_espacial.png")
]

for p in out_paths:
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"Mapa guardado en: {p}")

plt.close(fig)
print("generar_mapa_profesional.py finalizado con éxito.")
