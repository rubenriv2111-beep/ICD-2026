import os
import sys
import json
import time
import struct
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, Point, LineString, MultiLineString
from shapely.ops import transform
from pyproj import Transformer
import pyproj
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 80)
print(" SCRIPT REPRODUCIBLE: CONSTRUCCIÓN DEL DATASET Y PLIEGUES ESPACIALES")
print(" Proyecto: Rediseño Metodológico Cerro Prieto / Rift de Baja California")
print("=" * 80)

t0_total = time.time()

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
DATA_DIR = os.path.join(BASE_DIR, "Proyecto_Ciencia_De_Datos", "Notebook del proyecto", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
ART_DIR = r"C:\Users\ruben\.gemini\antigravity\brain\4b5923ef-c3b4-4cf6-868c-821d9273731f"

for d in [RESULTS_DIR, FIGURES_DIR, os.path.join(ART_DIR, "results"), os.path.join(ART_DIR, "figures")]:
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE LECTURA SHAPEFILE Y DBF
# -----------------------------------------------------------------------------
def read_dbf(dbf_path):
    with open(dbf_path, 'rb') as f:
        header = f.read(32)
        num_records, header_len, record_len = struct.unpack('<IHH', header[4:12])
        fields = []
        while True:
            first_byte = f.read(1)
            if not first_byte or first_byte == b'\r':
                break
            field_data = first_byte + f.read(31)
            name = field_data[:11].replace(b'\x00', b'').decode('latin1').strip()
            typ = chr(field_data[11])
            length = field_data[16]
            fields.append((name, typ, length))
        
        f.seek(header_len)
        records = []
        for _ in range(num_records):
            record_data = f.read(record_len)
            if len(record_data) < record_len:
                break
            offset = 1
            rec = {}
            for name, typ, length in fields:
                val = record_data[offset:offset+length].decode('latin1', errors='ignore').strip()
                rec[name] = val
                offset += length
            records.append(rec)
    return fields, records

def read_shp_lines(shp_path):
    with open(shp_path, 'rb') as f:
        header = f.read(100)
        file_len = struct.unpack('>I', header[24:28])[0] * 2
        geometries = []
        while f.tell() < file_len:
            rec_hdr = f.read(8)
            if len(rec_hdr) < 8:
                break
            rec_num, content_len = struct.unpack('>II', rec_hdr)
            content_bytes = f.read(content_len * 2)
            if len(content_bytes) < content_len * 2:
                break
            stype = struct.unpack('<I', content_bytes[:4])[0]
            if stype in (3, 13, 23):
                num_parts, num_points = struct.unpack('<II', content_bytes[36:44])
                parts = struct.unpack(f'<{num_parts}I', content_bytes[44:44+4*num_parts])
                pts_offset = 44 + 4 * num_parts
                pts_data = content_bytes[pts_offset:pts_offset + 16 * num_points]
                raw_pts = struct.unpack(f'<{2*num_points}d', pts_data)
                parts_indices = list(parts) + [num_points]
                lines = []
                for p_idx in range(num_parts):
                    p_start = parts_indices[p_idx]
                    p_end = parts_indices[p_idx+1]
                    line = [(raw_pts[2*i], raw_pts[2*i+1]) for i in range(p_start, p_end)]
                    lines.append(line)
                geometries.append(lines)
            else:
                geometries.append([])
    return geometries

# -----------------------------------------------------------------------------
# PARTE A. AUDITORÍA DE INSUMOS, DATUM Y FALLAS
# -----------------------------------------------------------------------------
print("\n[PARTE A1 & A2] Auditoría y reproyección de fallas estructurales...")

geojson_path = os.path.join(BASE_DIR, "Poligono_San_Felipe.geojson")
with open(geojson_path, "r", encoding="utf-8") as f:
    poly_data = json.load(f)

geom_wgs84 = shape(poly_data["features"][0]["geometry"])
trans_wgs_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
poly_utm = transform(trans_wgs_to_utm.transform, geom_wgs84)

# Reproyección explícita NAD27 -> WGS84 UTM 11N
tg = pyproj.transformer.TransformerGroup("EPSG:26711", "EPSG:32611")
trans_nad27_wgs84 = tg.transformers[1] # "Inverse of UTM zone 11N + NAD27 to WGS 84 (4) + UTM zone 11N"
trans_desc = trans_nad27_wgs84.description

mex_shp = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\mexicali_fallas\conjunto_de_datos\cdgv_1_250_000_serie1\i1112lg.shp")
mex_dbf = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\mexicali_fallas\conjunto_de_datos\cdgv_1_250_000_serie1\i1112lg.dbf")
sf_shp = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\san_felipe_fallas\conjunto_de_datos\cdgv_1_250_000_serie1\h1103lg.shp")
sf_dbf = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\san_felipe_fallas\conjunto_de_datos\cdgv_1_250_000_serie1\h1103lg.dbf")

fields_mex, recs_mex = read_dbf(mex_dbf)
geoms_mex = read_shp_lines(mex_shp)
fields_sf, recs_sf = read_dbf(sf_dbf)
geoms_sf = read_shp_lines(sf_shp)

all_fault_rows = []
all_reproj_lines = []
vertex_displacements = []

def process_fault_layer(geoms, recs, src_name, id_col):
    for i, (g_parts, rec) in enumerate(zip(geoms, recs)):
        reproj_parts = []
        for part in g_parts:
            reproj_part = []
            for x, y in part:
                xw, yw = trans_nad27_wgs84.transform(x, y)
                vertex_displacements.append(((xw - x)**2 + (yw - y)**2)**0.5)
                reproj_part.append((xw, yw))
            if len(reproj_part) >= 2:
                reproj_parts.append(LineString(reproj_part))
        if not reproj_parts:
            continue
        geom_shp = reproj_parts[0] if len(reproj_parts) == 1 else MultiLineString(reproj_parts)
        all_reproj_lines.append(geom_shp)
        dist_to_poly = poly_utm.distance(geom_shp)
        
        all_fault_rows.append({
            "capa": src_name,
            "fid": i + 1,
            "id_sgm": rec.get(id_col, ""),
            "entidad": rec.get("ENTIDAD", ""),
            "tipo": rec.get("TIPO", ""),
            "mov_hor": rec.get("MOV_HOR", ""),
            "mov_ver": rec.get("MOV_VER", ""),
            "representa": rec.get("REPRESENTA", ""),
            "longitud_m": round(float(rec.get("LENGTH", 0)), 2),
            "dist_a_poligono_m": round(dist_to_poly, 2),
            "dist_a_poligono_km": round(dist_to_poly / 1000.0, 3)
        })

process_fault_layer(geoms_mex, recs_mex, "i1112lg_Mexicali", "I1112LG_ID")
process_fault_layer(geoms_sf, recs_sf, "h1103lg_SanFelipe", "H1103LG_ID")

df_fallas = pd.DataFrame(all_fault_rows).sort_values("dist_a_poligono_m").reset_index(drop=True)
mean_vertex_disp = np.mean(vertex_displacements)

df_fallas.to_csv(os.path.join(RESULTS_DIR, "fallas_auditoria.csv"), index=False)
df_fallas.to_csv(os.path.join(ART_DIR, "results", "fallas_auditoria.csv"), index=False)

closest_fault = df_fallas.iloc[0]
print(f"Transformación aplicada: {trans_desc}")
print(f"Desplazamiento horizontal medio por cambio de datum: {mean_vertex_disp:.2f} m")
print(f"Traza más cercana: FID {closest_fault['fid']} ({closest_fault['entidad']} {closest_fault['tipo']}), distancia = {closest_fault['dist_a_poligono_m']:.2f} m ({closest_fault['dist_a_poligono_km']:.3f} km)")
print(f"Total trazas a <15 km: {(df_fallas['dist_a_poligono_km'] <= 15.0).sum()}")

fault_pts = []
for line in all_reproj_lines:
    sub_lines = [line] if line.geom_type == 'LineString' else list(line.geoms)
    for s_line in sub_lines:
        L = s_line.length
        n_pts = max(int(np.ceil(L / 15.0)), 2)
        sampled_pts = [s_line.interpolate(d).coords[0] for d in np.linspace(0, L, n_pts)]
        fault_pts.extend(sampled_pts)

tree_faults = cKDTree(np.array(fault_pts))
print(f"Puntos discretizados de fallas para cKDTree: {len(fault_pts):,}")

# -----------------------------------------------------------------------------
# PARTE A3, A4, A5, A6: DOMINIO VÁLIDO Y MÁSCARAS RÁSTER
# -----------------------------------------------------------------------------
print("\n[PARTE A3 & A4] Procesamiento de capas ráster y definición del dominio válido...")

elev_path = os.path.join(BASE_DIR, "elevaciones30.tif")
pend_path = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\pendientes30.tif")
mag_path = os.path.join(BASE_DIR, r"Procesamiento de imagenes\mapa_aeromagnetico_utm11n_continuo.tif")
lst_path = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\LC09_L2SP_038038_20260925_20260926_02_T1_ST_B10.TIF")
b4_path = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\LC09_L2SP_038038_20260925_20260926_02_T1_SR_B4.TIF")
b5_path = os.path.join(BASE_DIR, r"LC09_L2SP_038038_20260925_20260926_02_T1\LC09_L2SP_038038_20260925_20260926_02_T1_SR_B5.TIF")

with rasterio.open(elev_path) as src_e, rasterio.open(mag_path) as src_m, rasterio.open(lst_path) as src_l:
    inter_left = max(src_e.bounds.left, src_m.bounds.left, src_l.bounds.left)
    inter_bottom = max(src_e.bounds.bottom, src_m.bounds.bottom, src_l.bounds.bottom)
    inter_right = min(src_e.bounds.right, src_m.bounds.right, src_l.bounds.right)
    inter_top = min(src_e.bounds.top, src_m.bounds.top, src_l.bounds.top)

with rasterio.open(elev_path) as src_e:
    win = from_bounds(inter_left, inter_bottom, inter_right, inter_top, transform=src_e.transform)
    col_off = int(np.floor(win.col_off))
    row_off = int(np.floor(win.row_off))
    width = int(np.ceil(win.width))
    height = int(np.ceil(win.height))
    win = rasterio.windows.Window(col_off, row_off, width, height)
    win_transform = rasterio.windows.transform(win, src_e.transform)
    win_shape = (height, width)
    
    # Calibración física de elevación: elevaciones30.tif almacena valores divididos entre 11.1111 (escalados por 0.09)
    # Dividir entre 0.09 recupera exactamente la cota en metros sobre el nivel del mar (CEM 3.0 / MDE)
    dem_raw = src_e.read(1, window=win)
    dem = dem_raw / 0.09

with rasterio.open(pend_path) as src_p:
    slope = src_p.read(1, window=win)

def resample_window(path, nodata_val):
    with rasterio.open(path) as src:
        dest = np.empty(win_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=win_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata_val,
            dst_nodata=nodata_val
        )
        return dest

print("  Remuestreando capas continuas al grid común 30m...")
mag = resample_window(mag_path, -99999.0)
st10 = resample_window(lst_path, 0.0)
sr4 = resample_window(b4_path, 0.0)
sr5 = resample_window(b5_path, 0.0)

# Fórmulas físicas calibradas
# LST (°C): Landsat Collection 2 Kelvin - 273.15 = DN * 0.00341802 - 124.15
lst_celsius = np.where(st10 > 0, st10 * 0.00341802 - 124.15, np.nan)

# NDVI con factores oficiales de reflectancia superficial
sr4_refl = np.clip(np.where(sr4 > 0, sr4 * 0.0000275 - 0.2, 0.0), 0.0, 1.0)
sr5_refl = np.clip(np.where(sr5 > 0, sr5 * 0.0000275 - 0.2, 0.0), 0.0, 1.0)
denom_refl = sr5_refl + sr4_refl
ndvi_refl = np.where(denom_refl > 0, (sr5_refl - sr4_refl) / denom_refl, 0.0)
ndvi_refl = np.clip(ndvi_refl, -1.0, 1.0)

# NDVI directamente sobre DN crudo (para auditoría de continuidad contra el dataset original)
ndvi_raw_dn = np.where((sr5 + sr4) > 0, (sr5 - sr4) / (sr5 + sr4), 0.0)

# Máscara válida conjunta
mask_valid = (~np.isnan(dem)) & (dem > -100) & (mag > -50000) & (st10 > 0) & (sr4 > 0) & (sr5 > 0)

# Máscara del Golfo de California (Mar contiguo con elevación <= 0)
elev_zero = (dem <= 0.0)
labeled_water, _ = ndimage.label(elev_zero)
edge_ids = np.unique(np.concatenate([labeled_water[-1, :], labeled_water[:, -1]]))
edge_ids = edge_ids[edge_ids > 0]
edge_sizes = [(c, np.sum(labeled_water == c)) for c in edge_ids]
edge_sizes.sort(key=lambda x: x[1], reverse=True)
sea_component_id = edge_sizes[0][0]
mask_sea = (labeled_water == sea_component_id)

mask_candidates = mask_valid & (~mask_sea)

n_total_box = win_shape[0] * win_shape[1]
n_valid_domain = int(np.sum(mask_valid))
n_sea_excluded = int(np.sum(mask_sea & mask_valid))
n_candidates_land = int(np.sum(mask_candidates))

area_box_km2 = n_total_box * 900.0 / 1e6
area_valid_km2 = n_valid_domain * 900.0 / 1e6
area_sea_km2 = n_sea_excluded * 900.0 / 1e6
area_cand_km2 = n_candidates_land * 900.0 / 1e6

print(f"Celdas en rectángulo común: {n_total_box:,} ({area_box_km2:.2f} km2)")
print(f"Celdas válidas (todas las capas): {n_valid_domain:,} ({area_valid_km2:.2f} km2)")
print(f"Celdas de mar excluidas: {n_sea_excluded:,} ({area_sea_km2:.2f} km2)")
print(f"Dominio continental candidato: {n_candidates_land:,} ({area_cand_km2:.2f} km2)")

# -----------------------------------------------------------------------------
# PARTE B. MUESTREO DE NEGATIVOS Y ENSAMBLE DEL DATASET
# -----------------------------------------------------------------------------
print("\n[PARTE B] Ensamble de positivos y muestreo estratificado por bandas...")

csv_orig_path = os.path.join(DATA_DIR, "datos_geotermia_baja_california.csv")
df_orig = pd.read_csv(csv_orig_path)
df_pos_orig = df_orig[df_orig["Target_Geotermico"] == 1].copy().reset_index(drop=True)
N_POS = len(df_pos_orig)

# Rasterizar polígono sobre la cuadrícula para calcular distancias exactas
poly_mask = np.zeros(win_shape, dtype=bool)
rows_pos = np.int32(np.floor((win_transform.f - df_pos_orig["Y_UTM"].values) / 30.0))
cols_pos = np.int32(np.floor((df_pos_orig["X_UTM"].values - win_transform.c) / 30.0))
valid_pos_coords = (rows_pos >= 0) & (rows_pos < win_shape[0]) & (cols_pos >= 0) & (cols_pos < win_shape[1])
poly_mask[rows_pos[valid_pos_coords], cols_pos[valid_pos_coords]] = True

# Distancia euclidiana exacta en metros al polígono
dist_grid_m = ndimage.distance_transform_edt(~poly_mask, sampling=(30.0, 30.0))

# Clasificación de píxeles candidatos por banda
mask_b1_cand = mask_candidates & (dist_grid_m >= 500.0) & (dist_grid_m < 2000.0)
mask_b2_cand = mask_candidates & (dist_grid_m >= 2000.0) & (dist_grid_m < 5000.0)
mask_b3_cand = mask_candidates & (dist_grid_m >= 5000.0)
mask_excl = mask_candidates & (dist_grid_m > 0.0) & (dist_grid_m < 500.0)

n_b1_cand = int(np.sum(mask_b1_cand))
n_b2_cand = int(np.sum(mask_b2_cand))
n_b3_cand = int(np.sum(mask_b3_cand))
n_excl_cand = int(np.sum(mask_excl))

print(f"Celdas candidatas Banda 1 (0.5 - 2 km):  {n_b1_cand:,} ({n_b1_cand*900/1e6:.2f} km2)")
print(f"Celdas candidatas Banda 2 (2 - 5 km):    {n_b2_cand:,} ({n_b2_cand*900/1e6:.2f} km2)")
print(f"Celdas candidatas Banda 3 (> 5 km):      {n_b3_cand:,} ({n_b3_cand*900/1e6:.2f} km2)")
print(f"Zona de amortiguamiento (0 - 0.5 km):    {n_excl_cand:,} ({n_excl_cand*900/1e6:.2f} km2) [EXCLUIDA]")

N_SAMPLE_BAND = 21667

def sample_band_indices(mask, n_sample, seed):
    r, c = np.where(mask)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(r), size=n_sample, replace=False)
    return r[chosen], c[chosen]

r_b1, c_b1 = sample_band_indices(mask_b1_cand, N_SAMPLE_BAND, SEED)
r_b2, c_b2 = sample_band_indices(mask_b2_cand, N_SAMPLE_BAND, SEED + 1)
r_b3, c_b3 = sample_band_indices(mask_b3_cand, N_SAMPLE_BAND, SEED + 2)

def rc_to_utm(r, c):
    x = win_transform.c + (c + 0.5) * 30.0
    y = win_transform.f - (r + 0.5) * 30.0
    return x, y

x_b1, y_b1 = rc_to_utm(r_b1, c_b1)
x_b2, y_b2 = rc_to_utm(r_b2, c_b2)
x_b3, y_b3 = rc_to_utm(r_b3, c_b3)

df_neg_list = []
for name, r_idx, c_idx, x_coords, y_coords, n_cand in [
    ("B1", r_b1, c_b1, x_b1, y_b1, n_b1_cand),
    ("B2", r_b2, c_b2, x_b2, y_b2, n_b2_cand),
    ("B3", r_b3, c_b3, x_b3, y_b3, n_b3_cand)
]:
    df_band = pd.DataFrame({
        "x_utm": x_coords,
        "y_utm": y_coords,
        "y": 0,
        "banda": name,
        "dist_poligono_m": dist_grid_m[r_idx, c_idx],
        "prob_inclusion": N_SAMPLE_BAND / n_cand,
        "LST": lst_celsius[r_idx, c_idx],
        "NDVI": ndvi_refl[r_idx, c_idx],
        "NDVI_raw_dn": ndvi_raw_dn[r_idx, c_idx],
        "Elevaciones": dem[r_idx, c_idx],
        "Pendientes": slope[r_idx, c_idx],
        "Magnetometria": mag[r_idx, c_idx]
    })
    df_neg_list.append(df_band)

df_negs = pd.concat(df_neg_list, ignore_index=True)

df_pos = pd.DataFrame({
    "x_utm": df_pos_orig["X_UTM"].values,
    "y_utm": df_pos_orig["Y_UTM"].values,
    "y": 1,
    "banda": "POS",
    "dist_poligono_m": 0.0,
    "prob_inclusion": 1.0,
    "LST": lst_celsius[rows_pos, cols_pos],
    "NDVI": ndvi_refl[rows_pos, cols_pos],
    "NDVI_raw_dn": ndvi_raw_dn[rows_pos, cols_pos],
    "Elevaciones": dem[rows_pos, cols_pos],
    "Pendientes": slope[rows_pos, cols_pos],
    "Magnetometria": mag[rows_pos, cols_pos]
})

df_all = pd.concat([df_pos, df_negs], ignore_index=True)
df_all["id"] = np.arange(1, len(df_all) + 1)

# Proximidad a Fallas WGS84 para las 89,815 celdas
print("  Calculando dist_fallas_SGM_m para las 89,815 celdas...")
all_coords = df_all[["x_utm", "y_utm"]].values
dists_fallas, _ = tree_faults.query(all_coords)
df_all["dist_fallas_SGM_m"] = dists_fallas

# Coordenadas geográficas WGS84
trans_utm_to_wgs = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
lons, lats = trans_utm_to_wgs.transform(df_all["x_utm"].values, df_all["y_utm"].values)
df_all["lon"] = lons
df_all["lat"] = lats

# Heurísticas de interacción del manuscrito
df_all["Ratio_Termo_Estructural"] = df_all["LST"] / np.log(df_all["dist_fallas_SGM_m"] + 10.0)
df_all["Indice_Topo_Termico"] = df_all["LST"] / (df_all["Elevaciones"] + 50.0)

# Bandera de terreno: llanura
p99_elev_pos = np.percentile(df_all.loc[df_all["y"] == 1, "Elevaciones"], 99)
p99_slope_pos = np.percentile(df_all.loc[df_all["y"] == 1, "Pendientes"], 99)
df_all["llanura"] = ((df_all["Elevaciones"] <= p99_elev_pos) & (df_all["Pendientes"] <= p99_slope_pos)).astype(int)

print(f"Criterio llanura: Elevación <= {p99_elev_pos:.2f} m, Pendiente <= {p99_slope_pos:.2f}°")
for b in ["POS", "B1", "B2", "B3"]:
    pct_llan = (df_all[df_all["banda"] == b]["llanura"] == 1).mean() * 100
    print(f"  % Llanura en {b}: {pct_llan:.2f}%")

for b in ["POS", "B1", "B2", "B3"]:
    sub = df_all[df_all["banda"] == b]
    pct_elev_neg = (sub["Elevaciones"] <= 0).mean() * 100
    pct_ndvi_neg = (sub["NDVI"] < 0).mean() * 100
    print(f"  {b} -> Elev <= 0: {pct_elev_neg:.2f}%, NDVI < 0: {pct_ndvi_neg:.2f}%")

# -----------------------------------------------------------------------------
# CONTINUIDAD DE COVARIABLES (B2)
# -----------------------------------------------------------------------------
print("\n[PARTE B2] Evaluación de continuidad de covariables sobre los 24,814 positivos...")
df_pos_extracted = df_all[df_all["y"] == 1].copy().reset_index(drop=True)

cov_comp = []
for label, v_new, v_old, desc in [
    ("LST", df_pos_extracted["LST"].values, df_pos_orig["LST"].values, "Fórmula Collection 2 idéntica"),
    ("NDVI (Reflectancia Física)", df_pos_extracted["NDVI"].values, df_pos_orig["NDVI"].values, "Calibración física con escala y offset"),
    ("NDVI (Raw DN original)", df_pos_extracted["NDVI_raw_dn"].values, df_pos_orig["NDVI"].values, "Fórmula sobre DN sin calibrar"),
    ("Elevaciones", df_pos_extracted["Elevaciones"].values, df_pos_orig["Elevaciones"].values, "Recuperación de m.s.n.m. exactos (/0.09)"),
    ("Pendientes", df_pos_extracted["Pendientes"].values, df_pos_orig["Pendientes"].values, "Horn slope idéntico"),
    ("dist_fallas_SGM_m", df_pos_extracted["dist_fallas_SGM_m"].values, df_pos_orig["Prox_Fallas"].values, "Desfase por reproyección NAD27->WGS84"),
    ("Magnetometria (Continuo SGM)", df_pos_extracted["Magnetometria"].values, df_pos_orig["Magnetometria"].values, "Ráster continuo oficial SGM (r=1.000000)")
]:
    corr = np.corrcoef(v_new, v_old)[0, 1]
    mae = np.mean(np.abs(v_new - v_old))
    max_d = np.max(np.abs(v_new - v_old))
    cov_comp.append({
        "covariable": label,
        "correlacion_pearson": round(corr, 6),
        "diferencia_media_absoluta": round(mae, 4),
        "diferencia_maxima_absoluta": round(max_d, 4),
        "media_original": round(np.mean(v_old), 4),
        "media_nueva": round(np.mean(v_new), 4),
        "diagnostico": desc
    })
    print(f"  {label:30s} | Corr: {corr:9.6f} | MAE: {mae:9.4f} | Max Diff: {max_d:9.4f}")

df_continuidad = pd.DataFrame(cov_comp)
df_continuidad.to_csv(os.path.join(RESULTS_DIR, "continuidad_covariables.csv"), index=False)
df_continuidad.to_csv(os.path.join(ART_DIR, "results", "continuidad_covariables.csv"), index=False)

# Partición aleatoria estratificada (split_A) 70/15/15
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, temp_idx = next(sss1.split(df_all, df_all["banda"]))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
df_temp = df_all.iloc[temp_idx]
val_sub_idx, test_sub_idx = next(sss2.split(df_temp, df_temp["banda"]))
val_idx = temp_idx[val_sub_idx]
test_idx = temp_idx[test_sub_idx]

df_all["split_A"] = "train"
df_all.loc[val_idx, "split_A"] = "val"
df_all.loc[test_idx, "split_A"] = "test"

# Resumen de dominio y prevalencia
prevalencia_real = N_POS / n_valid_domain
prevalencia_dataset = N_POS / len(df_all)

df_dom_resumen = pd.DataFrame([
    {"metrica": "area_dominio_valido_km2", "valor": round(area_valid_km2, 4)},
    {"metrica": "celdas_dominio_valido", "valor": n_valid_domain},
    {"metrica": "area_b1_km2", "valor": round(n_b1_cand * 900 / 1e6, 4)},
    {"metrica": "celdas_b1_candidatas", "valor": n_b1_cand},
    {"metrica": "area_b2_km2", "valor": round(n_b2_cand * 900 / 1e6, 4)},
    {"metrica": "celdas_b2_candidatas", "valor": n_b2_cand},
    {"metrica": "area_b3_km2", "valor": round(n_b3_cand * 900 / 1e6, 4)},
    {"metrica": "celdas_b3_candidatas", "valor": n_b3_cand},
    {"metrica": "celdas_mar_excluidas", "valor": n_sea_excluded},
    {"metrica": "area_mar_excluida_km2", "valor": round(area_sea_km2, 4)},
    {"metrica": "positivos_censo_conteo", "valor": N_POS},
    {"metrica": "prevalencia_real", "valor": round(prevalencia_real, 6)},
    {"metrica": "prevalencia_real_pct", "valor": round(prevalencia_real * 100, 4)},
    {"metrica": "prevalencia_dataset", "valor": round(prevalencia_dataset, 6)},
    {"metrica": "prevalencia_dataset_pct", "valor": round(prevalencia_dataset * 100, 4)}
])
df_dom_resumen.to_csv(os.path.join(RESULTS_DIR, "dominio_resumen.csv"), index=False)
df_dom_resumen.to_csv(os.path.join(ART_DIR, "results", "dominio_resumen.csv"), index=False)

# -----------------------------------------------------------------------------
# PARTE C. PLIEGUES ESPACIALES (PROTOCOLO B) Y BUFFERS
# -----------------------------------------------------------------------------
print("\n[PARTE C] Generación de sectores espaciales (fold_B) y análisis de buffers...")

# C1: Positivos -> K-Means (K=5, seed=42)
km_pos = KMeans(n_clusters=5, random_state=SEED, n_init=10)
pos_coords = df_all.loc[df_all["y"] == 1, ["x_utm", "y_utm"]].values
pos_folds = km_pos.fit_predict(pos_coords) + 1
df_all.loc[df_all["y"] == 1, "fold_B"] = pos_folds

# Negativos B1 y B2 -> sector del positivo más cercano
tree_pos = cKDTree(pos_coords)
b12_mask = df_all["banda"].isin(["B1", "B2"])
b12_coords = df_all.loc[b12_mask, ["x_utm", "y_utm"]].values
_, nearest_pos_idx = tree_pos.query(b12_coords)
df_all.loc[b12_mask, "fold_B"] = pos_folds[nearest_pos_idx]

# Negativos B3 -> bloques de 10x10 km asignados al azar (seed 42)
b3_mask = (df_all["banda"] == "B3")
df_b3 = df_all.loc[b3_mask].copy()

b3_bx = (df_b3["x_utm"].values // 10000).astype(int)
b3_by = (df_b3["y_utm"].values // 10000).astype(int)
df_b3["block_id"] = [f"{x}_{y}" for x, y in zip(b3_bx, b3_by)]

unique_blocks = df_b3["block_id"].unique()
block_counts = df_b3["block_id"].value_counts()

rng_b3 = np.random.RandomState(SEED)
shuffled_blocks = rng_b3.permutation(unique_blocks)
fold_counts = {k: 0 for k in range(1, 6)}
block_to_fold = {}

sorted_blocks = sorted(shuffled_blocks, key=lambda b: block_counts[b], reverse=True)
for blk in sorted_blocks:
    cnt = block_counts[blk]
    best_f = min(fold_counts, key=fold_counts.get)
    block_to_fold[blk] = best_f
    fold_counts[best_f] += cnt

df_all.loc[b3_mask, "fold_B"] = df_b3["block_id"].map(block_to_fold).values
df_all["fold_B"] = df_all["fold_B"].astype(int)
df_all["fold5_asimetrico"] = (df_all["fold_B"] == 5).astype(int)

print("Distribución de fold_B por banda:")
print(pd.crosstab(df_all["banda"], df_all["fold_B"]))

# C2, C3, C4: Roles (train, test, buffer) para buffers {0, 500, 1000, 2000} m
buffers = [0, 500, 1000, 2000]
folds_k = [1, 2, 3, 4, 5]

counts_records = []
all_coords_array = df_all[["x_utm", "y_utm"]].values
roles_df_dict = {"id": df_all["id"].values, "fold_B": df_all["fold_B"].values, "banda": df_all["banda"].values, "y": df_all["y"].values, "fold5_asimetrico": df_all["fold5_asimetrico"].values}

for b in buffers:
    print(f"  Procesando buffer de {b} metros...")
    for k in folds_k:
        test_mask = (df_all["fold_B"] == k).values
        other_mask = ~test_mask
        
        test_indices = np.where(test_mask)[0]
        other_indices = np.where(other_mask)[0]
        
        test_pts_k = all_coords_array[test_indices]
        other_pts_k = all_coords_array[other_indices]
        
        tree_test = cKDTree(test_pts_k)
        tree_other = cKDTree(other_pts_k)
        
        if b == 0:
            is_buf_other = np.zeros(len(other_pts_k), dtype=bool)
        else:
            close_lists = tree_test.query_ball_tree(tree_other, r=b)
            flat_close = [idx for sub in close_lists for idx in sub]
            close_indices = np.unique(flat_close) if len(flat_close) > 0 else np.array([], dtype=int)
            is_buf_other = np.zeros(len(other_pts_k), dtype=bool)
            if len(close_indices) > 0:
                is_buf_other[close_indices] = True
        
        is_train_other = ~is_buf_other
        
        if is_train_other.sum() > 0 and b > 0:
            min_dist_to_test = tree_test.query(other_pts_k[is_train_other])[0].min()
            assert min_dist_to_test > b, f"Fallo de fuga: min dist {min_dist_to_test} <= {b}"
        else:
            min_dist_to_test = np.nan
        
        df_sub_test = df_all.iloc[test_indices]
        df_sub_other = df_all.iloc[other_indices]
        df_sub_buf = df_sub_other.iloc[is_buf_other]
        df_sub_train = df_sub_other.iloc[is_train_other]
        
        for b_name in ["POS", "B1", "B2", "B3", "TOTAL"]:
            if b_name == "TOTAL":
                n_test = len(df_sub_test)
                n_train = len(df_sub_train)
                n_buf = len(df_sub_buf)
                pos_train = (df_sub_train["y"] == 1).sum()
                pos_buf = (df_sub_buf["y"] == 1).sum()
                pos_test = (df_sub_test["y"] == 1).sum()
            else:
                n_test = (df_sub_test["banda"] == b_name).sum()
                n_train = (df_sub_train["banda"] == b_name).sum()
                n_buf = (df_sub_buf["banda"] == b_name).sum()
                pos_train = ((df_sub_train["banda"] == b_name) & (df_sub_train["y"] == 1)).sum()
                pos_buf = ((df_sub_buf["banda"] == b_name) & (df_sub_buf["y"] == 1)).sum()
                pos_test = ((df_sub_test["banda"] == b_name) & (df_sub_test["y"] == 1)).sum()
            
            pct_pos_lost = (pos_buf / (pos_train + pos_buf) * 100) if (pos_train + pos_buf) > 0 else 0.0
            es_inviable = (pos_test < 1000) or (pct_pos_lost > 50.0)
            
            counts_records.append({
                "buffer_m": b,
                "pliegue_test": k,
                "banda": b_name,
                "n_test": n_test,
                "n_train": n_train,
                "n_buffer": n_buf,
                "positivos_test": pos_test,
                "positivos_train": pos_train,
                "positivos_buffer": pos_buf,
                "pct_positivos_train_perdidos": round(pct_pos_lost, 2),
                "min_dist_train_test_m": round(min_dist_to_test, 2) if not np.isnan(min_dist_to_test) else 0.0,
                "estado_viabilidad": "Inviable" if es_inviable else "Viable"
            })
        
        # Guardar roles para b <= 1000
        if b in [0, 500, 1000]:
            col_name = f"role_b{b}_f{k}"
            role_arr = np.empty(len(df_all), dtype=object)
            role_arr[test_indices] = "test"
            role_arr[other_indices[is_buf_other]] = "buffer"
            role_arr[other_indices[~is_buf_other]] = "train"
            roles_df_dict[col_name] = role_arr

df_counts_buffer = pd.DataFrame(counts_records)
df_counts_buffer.to_csv(os.path.join(RESULTS_DIR, "conteos_pliegues_bandas_buffer.csv"), index=False)
df_counts_buffer.to_csv(os.path.join(ART_DIR, "results", "conteos_pliegues_bandas_buffer.csv"), index=False)
print("Saved conteos_pliegues_bandas_buffer.csv")

df_roles = pd.DataFrame(roles_df_dict)
df_roles.to_parquet(os.path.join(RESULTS_DIR, "pliegues_roles.parquet"), index=False)
df_roles.to_parquet(os.path.join(ART_DIR, "results", "pliegues_roles.parquet"), index=False)
print("Saved pliegues_roles.parquet")

# C5: Prevalencias basales
print("\n[PARTE C5] Prevalencias basales (PR-AUC baseline):")
for k in folds_k:
    sub_k = df_all[df_all["fold_B"] == k]
    p_k = (sub_k["y"] == 1).sum()
    n_b1 = (sub_k["banda"] == "B1").sum()
    n_b2 = (sub_k["banda"] == "B2").sum()
    n_b3 = (sub_k["banda"] == "B3").sum()
    prev_b1 = p_k / (p_k + n_b1)
    prev_b2 = p_k / (p_k + n_b2)
    prev_b3 = p_k / (p_k + n_b3)
    prev_all = p_k / len(sub_k)
    print(f"  Fold {k}: Pos={p_k} | B1={n_b1} (prev={prev_b1:.4f}) | B2={n_b2} (prev={prev_b2:.4f}) | B3={n_b3} (prev={prev_b3:.4f}) | Total={len(sub_k)} (prev={prev_all:.4f})")

# -----------------------------------------------------------------------------
# PARTE D. DIAGNÓSTICO DE SOLAPAMIENTO
# -----------------------------------------------------------------------------
print("\n[PARTE D] Diagnóstico de solapamiento físico y prueba KS...")
covariates_list = ["LST", "NDVI", "Elevaciones", "Pendientes", "dist_fallas_SGM_m", "Magnetometria"]
solap_records = []
df_pos_final = df_all[df_all["y"] == 1]

for cov in covariates_list:
    v_pos = df_pos_final[cov].values
    pos_p1 = np.percentile(v_pos, 1)
    pos_p5 = np.percentile(v_pos, 5)
    pos_med = np.median(v_pos)
    pos_p95 = np.percentile(v_pos, 95)
    pos_p99 = np.percentile(v_pos, 99)
    
    eval_groups = [
        ("B1", df_all[df_all["banda"] == "B1"]),
        ("B2", df_all[df_all["banda"] == "B2"]),
        ("B3_Total", df_all[df_all["banda"] == "B3"]),
        ("B3_Llanura", df_all[(df_all["banda"] == "B3") & (df_all["llanura"] == 1)]),
        ("B3_NoLlanura", df_all[(df_all["banda"] == "B3") & (df_all["llanura"] == 0)])
    ]
    
    for grp_name, df_grp in eval_groups:
        v_neg = df_grp[cov].values
        neg_p5 = np.percentile(v_neg, 5)
        neg_med = np.median(v_neg)
        neg_p95 = np.percentile(v_neg, 95)
        
        in_range = ((v_neg >= pos_p1) & (v_neg <= pos_p99)).mean() * 100
        ks_stat, ks_pval = ks_2samp(v_pos, v_neg)
        
        solap_records.append({
            "covariable": cov,
            "grupo_negativo": grp_name,
            "n_negativos": len(v_neg),
            "pos_p5": round(pos_p5, 4),
            "pos_mediana": round(pos_med, 4),
            "pos_p95": round(pos_p95, 4),
            "neg_p5": round(neg_p5, 4),
            "neg_mediana": round(neg_med, 4),
            "neg_p95": round(neg_p95, 4),
            "pct_neg_en_p1_p99_pos": round(in_range, 2),
            "ks_estadistico_D": round(ks_stat, 4),
            "ks_pvalor": ks_pval
        })

df_solap = pd.DataFrame(solap_records)
df_solap.to_csv(os.path.join(RESULTS_DIR, "solapamiento_bandas.csv"), index=False)
df_solap.to_csv(os.path.join(ART_DIR, "results", "solapamiento_bandas.csv"), index=False)
print("Saved solapamiento_bandas.csv")

# Guardar dataset final en formato parquet
parquet_path = os.path.join(RESULTS_DIR, "dataset_final.parquet")
df_all.to_parquet(parquet_path, index=False)
df_all.to_parquet(os.path.join(ART_DIR, "results", "dataset_final.parquet"), index=False)
print(f"Saved dataset_final.parquet ({len(df_all):,} filas, {len(df_all.columns)} columnas)")

# -----------------------------------------------------------------------------
# FIGURA: MAPA DE PLIEGUES Y BUFFER (500 M)
# -----------------------------------------------------------------------------
print("\n[FIGURA] Generando mapa_pliegues_buffer.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=300)

to_km = 1e-3
colors_folds = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Panel A: Dominio regional y pliegues espaciales
for k in range(1, 6):
    mask_k = (df_all["fold_B"] == k) & (df_all["y"] == 0)
    sub = df_all[mask_k].sample(n=min(mask_k.sum(), 4000), random_state=SEED)
    ax1.scatter(sub["x_utm"]*to_km, sub["y_utm"]*to_km, c=colors_folds[k-1], s=4, alpha=0.45, label=f"Pliegue {k} (Negativos)")

ax1.scatter(df_all.loc[df_all["y"] == 1, "x_utm"]*to_km, df_all.loc[df_all["y"] == 1, "y_utm"]*to_km, c='black', s=6, alpha=0.8, label="Cerro Prieto (Positivos)")

minx, miny, maxx, maxy = poly_utm.buffer(6000).bounds
rect_zoom = plt.Rectangle(
    (minx*to_km, miny*to_km), (maxx - minx)*to_km, (maxy - miny)*to_km,
    linewidth=2, edgecolor='#e63946', facecolor='none', linestyle='--', label="Ventana Local (Panel B)"
)
ax1.add_patch(rect_zoom)

ax1.set_title("(a) Dominio Regional y 5 Pliegues Espaciales (Protocolo B)", fontsize=13, weight='bold')
ax1.set_xlabel("UTM Este (km)", fontsize=11)
ax1.set_ylabel("UTM Norte (km)", fontsize=11)
ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
ax1.grid(True, linestyle=":", alpha=0.6)

# Panel B: Detalle Cerro Prieto con buffer de 500 m
sub_zoom = df_all[(df_all["x_utm"] >= minx) & (df_all["x_utm"] <= maxx) & (df_all["y_utm"] >= miny) & (df_all["y_utm"] <= maxy)]

for k in range(1, 6):
    mask_pos_k = (sub_zoom["y"] == 1) & (sub_zoom["fold_B"] == k)
    if mask_pos_k.sum() > 0:
        ax2.scatter(sub_zoom.loc[mask_pos_k, "x_utm"]*to_km, sub_zoom.loc[mask_pos_k, "y_utm"]*to_km, c=colors_folds[k-1], s=12, label=f"Positivos Sector {k}")

# Mostrar buffer de aislamiento de 500m (cuando Pliegue 1 es Test)
mask_buf_f1 = (df_roles["role_b500_f1"] == "buffer").values
sub_buf = df_all.iloc[mask_buf_f1]
sub_buf_zoom = sub_buf[(sub_buf["x_utm"] >= minx) & (sub_buf["x_utm"] <= maxx) & (sub_buf["y_utm"] >= miny) & (sub_buf["y_utm"] <= maxy)]
ax2.scatter(sub_buf_zoom["x_utm"]*to_km, sub_buf_zoom["y_utm"]*to_km, c='gray', s=8, alpha=0.4, label="Zona Buffer 500m (Excluida en Train)")

px, py = poly_utm.exterior.xy
ax2.plot(np.array(px)*to_km, np.array(py)*to_km, color='black', linewidth=2.0, label="Polígono Concesión")

ax2.set_xlim(minx*to_km, maxx*to_km)
ax2.set_ylim(miny*to_km, maxy*to_km)
ax2.set_title("(b) Cerro Prieto: 5 Sectores y Aislamiento por Buffer de 500 m", fontsize=13, weight='bold')
ax2.set_xlabel("UTM Este (km)", fontsize=11)
ax2.set_ylabel("UTM Norte (km)", fontsize=11)
ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "mapa_pliegues_buffer.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(ART_DIR, "figures", "mapa_pliegues_buffer.png"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig_path}")

t_end = time.time()
print(f"\nEJECUCIÓN COMPLETADA EXITOSAMENTE EN {t_end - t0_total:.2f} SEGUNDOS.")
