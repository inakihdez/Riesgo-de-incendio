"""
procesar_incendio.py
--------------------
Descarga el ZIP de peligro de incendio forestal de AEMET/MITECO,
cruza los GeoTIFF con los polígonos municipales del IGN y genera:
  - CSV para Datawrapper (fechas desde hoy)
  - PNG del raster del día actual para el mapa Leaflet
  - JSON de metadatos del mapa (bounds, fecha)

Requiere:
    pip install requests rasterio geopandas scipy numpy Pillow

Variables de entorno:
    DW_API_KEY   -> API key de Datawrapper
    DW_CHART_ID  -> ID visualización (por defecto: OU4ZS)
"""

import os, re, tarfile, tempfile, csv, logging, json
from datetime import datetime, timedelta
from pathlib import Path

import requests
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy import stats
from PIL import Image

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

AEMET_URL    = "https://www.aemet.es/es/api-eltiempo/incendios/download"
SHP_PEN      = Path("shp/recintos_municipales_inspire_peninbal_etrs89.shp")
SHP_CAN      = Path("shp/recintos_municipales_inspire_canarias_regcan95.shp")
GEOJSON_PATH = Path("municipios.geojson")
OUTPUT_CSV   = Path("output/riesgo_incendio_municipios.csv")
OUTPUT_PNG   = Path("output/mapa_hoy.png")       # raster del día actual
OUTPUT_META  = Path("output/mapa_meta.json")     # bounds + fecha para Leaflet

# Paleta de colores MITECO (valor 0=transparente, 1-6=escala)
COLORES = {
    0: (0,   0,   0,   0),    # transparente (sin dato / urbano)
    1: (75,  150, 227, 200),  # Muy bajo  — azul
    2: (81,  209, 246, 200),  # Bajo      — celeste
    3: (87,  229, 32,  200),  # Moderado  — verde
    4: (249, 251, 47,  200),  # Alto      — amarillo
    5: (239, 133, 4,   200),  # Muy alto  — naranja
    6: (245, 35,  0,   200),  # Extremo   — rojo
}

RISK_LABELS = {
    1: "Muy bajo", 2: "Bajo",     3: "Moderado",
    4: "Alto",     5: "Muy alto", 6: "Extremo",
}

PROV_NAMES = {
    "01":"Álava",      "02":"Albacete",    "03":"Alicante",   "04":"Almería",
    "05":"Ávila",      "06":"Badajoz",     "07":"Baleares",   "08":"Barcelona",
    "09":"Burgos",     "10":"Cáceres",     "11":"Cádiz",      "12":"Castellón",
    "13":"Ciudad Real","14":"Córdoba",     "15":"A Coruña",   "16":"Cuenca",
    "17":"Girona",     "18":"Granada",     "19":"Guadalajara","20":"Guipúzcoa",
    "21":"Huelva",     "22":"Huesca",      "23":"Jaén",       "24":"León",
    "25":"Lleida",     "26":"La Rioja",    "27":"Lugo",       "28":"Madrid",
    "29":"Málaga",     "30":"Murcia",      "31":"Navarra",    "32":"Ourense",
    "33":"Asturias",   "34":"Palencia",    "35":"Las Palmas", "36":"Pontevedra",
    "37":"Salamanca",  "38":"Tenerife",    "39":"Cantabria",  "40":"Segovia",
    "41":"Sevilla",    "42":"Soria",       "43":"Tarragona",  "44":"Teruel",
    "45":"Toledo",     "46":"Valencia",    "47":"Valladolid", "48":"Vizcaya",
    "49":"Zamora",     "50":"Zaragoza",    "51":"Ceuta",      "52":"Melilla",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Descarga y extracción de TIFs
# ---------------------------------------------------------------------------

def descargar_tifs(url: str, destino: Path) -> Path:
    log.info("Descargando datos de AEMET/MITECO...")
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    destino.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp_path) as tar:
        tar.extractall(destino)
    tmp_path.unlink()
    return destino


def detectar_fecha_y_tifs(carpeta: Path):
    tifs = list(carpeta.rglob("down_*_peligro_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No se encontraron TIF en {carpeta}")
    m = re.search(r"down_(\d{8})_", tifs[0].name)
    if not m:
        raise ValueError(f"Nombre inesperado: {tifs[0].name}")
    fecha_base = datetime.strptime(m.group(1), "%Y%m%d")
    tif_map = {}
    for tif in tifs:
        m2 = re.search(r"peligro_([pc])_D0(\d)", tif.name)
        if m2:
            tif_map[(m2.group(1), int(m2.group(2)))] = tif
    log.info(f"Fecha base: {fecha_base.date()}  |  TIFs: {len(tif_map)}")
    return fecha_base, tif_map


# ---------------------------------------------------------------------------
# 2. Municipios
# ---------------------------------------------------------------------------

def construir_geojson(shp_pen, shp_can, geojson_path):
    log.info("Generando municipios.geojson desde shapefiles IGN...")
    pen = gpd.read_file(shp_pen).to_crs("EPSG:4326")
    can = gpd.read_file(shp_can).to_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame(gpd.pd.concat([pen, can], ignore_index=True), crs="EPSG:4326")
    gdf["prov_code"] = gdf["NATCODE"].str[6:8]
    gdf["provincia"] = gdf["prov_code"].map(PROV_NAMES)
    gdf = gdf[["NAMEUNIT", "provincia", "geometry"]].copy()
    gdf.columns = ["nombre", "provincia", "geometry"]
    gdf["geometry"] = gdf["geometry"].simplify(0.001, preserve_topology=True)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(geojson_path, driver="GeoJSON")
    log.info(f"municipios.geojson: {len(gdf)} municipios")
    return gdf


def cargar_municipios():
    if GEOJSON_PATH.exists():
        log.info("Cargando municipios.geojson (caché)...")
        return gpd.read_file(GEOJSON_PATH)
    return construir_geojson(SHP_PEN, SHP_CAN, GEOJSON_PATH)


# ---------------------------------------------------------------------------
# 3. Riesgo por polígono
# ---------------------------------------------------------------------------

def extraer_riesgo(geom, tif_path):
    if tif_path is None:
        return None
    try:
        with rasterio.open(tif_path) as src:
            b = src.bounds
            gb = geom.bounds
            if gb[2] < b.left or gb[0] > b.right or gb[3] < b.bottom or gb[1] > b.top:
                return None
            out, _ = rio_mask(src, [geom.__geo_interface__], crop=True, nodata=np.nan)
            data = out[0]
            valid = data[(data > 0) & ~np.isnan(data)]
            if len(valid) == 0:
                return None
            return int(stats.mode(valid, keepdims=True).mode[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. CSV para Datawrapper (siempre desde D00 = hoy)
# ---------------------------------------------------------------------------

def generar_csv(gdf, tif_map, fecha_base, output, start_offset=0):
    """
    start_offset: índice D en el que empieza "hoy" (calculado en main()
    comparando la fecha real con fecha_base, ya que AEMET no siempre
    etiqueta D00 como el día de hoy).
    """
    n_dias = max(d for (_, d) in tif_map.keys()) + 1
    fechas = []
    for d in range(start_offset, n_dias):
        fecha = (fecha_base + timedelta(days=d)).strftime("%d/%m")
        fechas.append(f"Hoy {fecha}" if d == start_offset else fecha)

    output.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    total = len(gdf)

    for i, row in gdf.iterrows():
        nombre = row["nombre"]
        prov   = row["provincia"] or ""
        geom   = row.geometry
        vals = []
        for d in range(start_offset, n_dias):
            v = (extraer_riesgo(geom, tif_map.get(("p", d))) or
                 extraer_riesgo(geom, tif_map.get(("c", d))) or 1)
            vals.append(v)
        out_row = {"Municipio": f"{nombre} ({prov})" if prov else nombre}
        for fecha, v in zip(fechas, vals):
            out_row[fecha] = RISK_LABELS.get(v, "")
        filas.append(out_row)
        if i % 500 == 0:
            log.info(f"  {i}/{total} municipios procesados...")

    filas.sort(key=lambda r: r["Municipio"])
    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)
    log.info(f"CSV: {output}  ({len(filas)} municipios, desde D{start_offset:02d})")


# ---------------------------------------------------------------------------
# 5. PNG del raster del día actual para Leaflet
# ---------------------------------------------------------------------------

def generar_png_mapa(tif_pen, tif_can, output_png, output_meta, fecha_base):
    """
    Combina los dos rasters (Península+Baleares y Canarias) en un
    PNG con transparencia usando la paleta de colores del MITECO.
    Guarda también un JSON con los bounds para Leaflet.
    """
    log.info("Generando PNG del mapa del día actual...")

    def tif_a_rgba(tif_path):
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            bounds = src.bounds
            h, w = data.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            for val, color in COLORES.items():
                mask = (data == val) if val > 0 else (data <= 0) | np.isnan(data)
                rgba[mask] = color
            return rgba, bounds

    rgba_pen, bounds_pen = tif_a_rgba(tif_pen)
    rgba_can, bounds_can = tif_a_rgba(tif_can)

    # Crear canvas combinado que englobe ambos rasters
    # Resolución: 0.01 grados por píxel (igual que los TIF)
    res = 0.01
    west  = min(bounds_pen.left,   bounds_can.left)
    east  = max(bounds_pen.right,  bounds_can.right)
    south = min(bounds_pen.bottom, bounds_can.bottom)
    north = max(bounds_pen.top,    bounds_can.top)

    width  = int(round((east  - west)  / res))
    height = int(round((north - south) / res))
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    def pegar(rgba, bounds):
        col0 = int(round((bounds.left   - west)  / res))
        row0 = int(round((north - bounds.top)     / res))
        h, w = rgba.shape[:2]
        r1, r2 = row0, row0 + h
        c1, c2 = col0, col0 + w
        # Solo pegar donde hay dato (alpha > 0)
        mask = rgba[:, :, 3] > 0
        canvas[r1:r2, c1:c2][mask] = rgba[mask]

    pegar(rgba_pen, bounds_pen)
    pegar(rgba_can, bounds_can)

    # ── Reproyección a Web Mercator (EPSG:3857) ──────────────────────────
    # El PNG está en WGS84 (grados). MapLibre/Leaflet dibujan en Web
    # Mercator, y superponer una imagen en grados sobre un mapa en Mercator
    # usando solo 4 esquinas produce una distorsión progresiva norte-sur
    # (más visible cuanto más lejos del ecuador). Reproyectamos el canvas
    # completo a Mercator para que el encaje sea exacto.
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import from_origin, array_bounds
    from pyproj import Transformer

    src_transform = from_origin(west, north, res, res)
    src_crs = "EPSG:4326"
    dst_crs = "EPSG:3857"

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, width, height,
        left=west, bottom=south, right=east, top=north,
    )

    canvas_merc = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    for b in range(4):
        reproject(
            source=canvas[:, :, b],
            destination=canvas_merc[:, :, b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,  # nearest: son colores categóricos, no continuos
        )

    # Bounds del canvas reproyectado, en metros Mercator
    merc_left, merc_bottom, merc_right, merc_top = array_bounds(dst_h, dst_w, dst_transform)

    # Convertir esas 4 esquinas de vuelta a lon/lat (WGS84) para pasárselas
    # a MapLibre — MapLibre las volverá a proyectar a Mercator internamente,
    # y como la imagen ya está espaciada uniformemente en Mercator, el
    # encaje será exacto en vez de aproximado.
    transformer = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    lon_w, lat_s = transformer.transform(merc_left,  merc_bottom)
    lon_e, lat_n = transformer.transform(merc_right, merc_top)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas_merc, "RGBA").save(output_png, optimize=True)
    log.info(f"PNG guardado (Web Mercator): {output_png}  ({output_png.stat().st_size // 1024} KB)")

    # Metadatos para Leaflet — bounds ya corregidos para encajar sin distorsión
    meta = {
        "fecha": fecha_base.strftime("%d/%m/%Y"),
        "bounds": [[lat_s, lon_w], [lat_n, lon_e]],
        "png": "mapa_hoy.png",
    }
    with open(output_meta, "w") as f:
        json.dump(meta, f)
    log.info(f"Metadatos: {output_meta}")


# ---------------------------------------------------------------------------
# 6. Subida a Datawrapper
# ---------------------------------------------------------------------------

def subir_a_datawrapper(csv_path):
    api_key  = os.getenv("DW_API_KEY")
    chart_id = os.getenv("DW_CHART_ID", "OU4ZS")
    if not api_key:
        log.info("DW_API_KEY no definido — se omite la subida.")
        return
    log.info(f"Subiendo a Datawrapper (chart {chart_id})...")
    with open(csv_path, encoding="utf-8-sig") as f:
        csv_text = f.read()
    r = requests.put(
        f"https://api.datawrapper.de/v3/charts/{chart_id}/data",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "text/csv"},
        data=csv_text.encode("utf-8"),
        timeout=60,
    )
    if r.ok:
        log.info("Datos subidos a Datawrapper.")
    else:
        log.error(f"Error Datawrapper: {r.status_code} {r.text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gdf = cargar_municipios()

    with tempfile.TemporaryDirectory() as tmpdir:
        tif_dir = descargar_tifs(AEMET_URL, Path(tmpdir) / "tifs")
        fecha_base, tif_map = detectar_fecha_y_tifs(tif_dir)
        generar_csv(gdf, tif_map, fecha_base, OUTPUT_CSV)

        n_dias = max(d for (_, d) in tif_map.keys()) + 1

        # AEMET no siempre etiqueta D00 como "hoy" — a veces el paquete se
        # genera de madrugada con la fecha del día que termina, y D01 es
        # el que corresponde al día real. Calculamos el offset comparando
        # la fecha real (hora de Madrid) con la fecha_base del paquete.
        try:
            from zoneinfo import ZoneInfo
            hoy_madrid = datetime.now(ZoneInfo("Europe/Madrid")).date()
        except Exception:
            hoy_madrid = datetime.utcnow().date()

        offset = (hoy_madrid - fecha_base.date()).days

        if 0 <= offset < n_dias:
            log.info(f"Hoy es {hoy_madrid} → usando D{offset:02d} como día actual "
                      f"(fecha_base del paquete: {fecha_base.date()}).")
        else:
            log.warning(
                f"⚠ Offset calculado ({offset}) fuera de rango [0,{n_dias-1}] — "
                f"hoy={hoy_madrid}, fecha_base={fecha_base.date()}. Usando D00 por defecto."
            )
            offset = 0

        # CSV para Datawrapper, empezando desde el día real
        generar_csv(gdf, tif_map, fecha_base, OUTPUT_CSV, start_offset=offset)

        # PNG para Leaflet del día real
        tif_pen = tif_map.get(("p", offset))
        tif_can = tif_map.get(("c", offset))
        if tif_pen and tif_can:
            fecha_hoy = fecha_base + timedelta(days=offset)
            generar_png_mapa(tif_pen, tif_can, OUTPUT_PNG, OUTPUT_META, fecha_hoy)
        else:
            log.warning("No se encontraron TIFs para el día actual.")

    subir_a_datawrapper(OUTPUT_CSV)
    log.info("Proceso completado.")
