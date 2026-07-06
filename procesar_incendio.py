"""
procesar_incendio.py
--------------------
Descarga el ZIP de peligro de incendio forestal de AEMET/MITECO,
cruza los GeoTIFF con los polígonos municipales del IGN y genera
un CSV que sube automáticamente a Datawrapper.

Uso:
    python procesar_incendio.py

Requiere:
    pip install requests rasterio geopandas scipy numpy

Variables de entorno necesarias para la subida a Datawrapper:
    DW_API_KEY   -> API key de Datawrapper (datawrapper.de/account/api-tokens)
    DW_CHART_ID  -> ID de la visualización (aparece en la URL al editar, ej: EqqfeFodYj)
"""

import os
import re
import tarfile
import tempfile
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy import stats

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

AEMET_URL = "https://www.aemet.es/es/api-eltiempo/incendios/download"

# Shapefiles del IGN — se asumen descargados y en esta ruta relativa.
# Descarga en: https://centrodedescargas.cnig.es/CentroDescargas/
SHP_PEN = Path("shp/recintos_municipales_inspire_peninbal_etrs89.shp")
SHP_CAN = Path("shp/recintos_municipales_inspire_canarias_regcan95.shp")

OUTPUT_CSV = Path("output/riesgo_incendio_municipios.csv")

RISK_LABELS = {
    1: "Muy bajo",
    2: "Bajo",
    3: "Moderado",
    4: "Alto",
    5: "Muy alto",
    6: "Extremo",
}

PROV_NAMES = {
    "01": "Álava", "02": "Albacete", "03": "Alicante", "04": "Almería",
    "05": "Ávila", "06": "Badajoz", "07": "Baleares", "08": "Barcelona",
    "09": "Burgos", "10": "Cáceres", "11": "Cádiz", "12": "Castellón",
    "13": "Ciudad Real", "14": "Córdoba", "15": "A Coruña", "16": "Cuenca",
    "17": "Girona", "18": "Granada", "19": "Guadalajara", "20": "Guipúzcoa",
    "21": "Huelva", "22": "Huesca", "23": "Jaén", "24": "León",
    "25": "Lleida", "26": "La Rioja", "27": "Lugo", "28": "Madrid",
    "29": "Málaga", "30": "Murcia", "31": "Navarra", "32": "Ourense",
    "33": "Asturias", "34": "Palencia", "35": "Las Palmas", "36": "Pontevedra",
    "37": "Salamanca", "38": "Tenerife", "39": "Cantabria", "40": "Segovia",
    "41": "Sevilla", "42": "Soria", "43": "Tarragona", "44": "Teruel",
    "45": "Toledo", "46": "Valencia", "47": "Valladolid", "48": "Vizcaya",
    "49": "Zamora", "50": "Zaragoza", "51": "Ceuta", "52": "Melilla",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Descarga y extracción
# ---------------------------------------------------------------------------

def descargar_zip(url: str, destino: Path) -> Path:
    """Descarga el tar.gz de AEMET y lo extrae en destino."""
    log.info("Descargando datos de AEMET/MITECO...")
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    log.info(f"Extrayendo en {destino}...")
    destino.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp_path) as tar:
        tar.extractall(destino)
    tmp_path.unlink()

    return destino


def detectar_fecha_y_tifs(carpeta: Path):
    """
    Detecta la fecha base y devuelve un dict:
        {('p'|'c', day_offset): Path}
    a partir de los TIF encontrados.
    """
    tifs = list(carpeta.rglob("down_*_peligro_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No se encontraron TIF en {carpeta}")

    # Extraer fecha del primer nombre: down_YYYYMMDD_peligro_p_D00.tif
    m = re.search(r"down_(\d{8})_", tifs[0].name)
    if not m:
        raise ValueError(f"Nombre de archivo inesperado: {tifs[0].name}")
    fecha_base = datetime.strptime(m.group(1), "%Y%m%d")

    tif_map = {}
    for tif in tifs:
        m2 = re.search(r"peligro_([pc])_D0(\d)", tif.name)
        if m2:
            zona = m2.group(1)   # 'p' = península, 'c' = canarias
            dia = int(m2.group(2))
            tif_map[(zona, dia)] = tif

    log.info(f"Fecha base: {fecha_base.date()}  |  TIFs encontrados: {len(tif_map)}")
    return fecha_base, tif_map


# ---------------------------------------------------------------------------
# 2. Carga de municipios IGN
# ---------------------------------------------------------------------------

def cargar_municipios(shp_pen: Path, shp_can: Path) -> gpd.GeoDataFrame:
    log.info("Cargando shapefiles del IGN...")
    pen = gpd.read_file(shp_pen).to_crs("EPSG:4326")
    can = gpd.read_file(shp_can).to_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame(
        gpd.pd.concat([pen, can], ignore_index=True),
        crs="EPSG:4326",
    )
    gdf["prov_code"] = gdf["NATCODE"].str[6:8]
    gdf["provincia"] = gdf["prov_code"].map(PROV_NAMES)
    log.info(f"Municipios cargados: {len(gdf)}")
    return gdf


# ---------------------------------------------------------------------------
# 3. Extracción de riesgo por polígono
# ---------------------------------------------------------------------------

def extraer_riesgo(geom, tif_path: Path) -> int | None:
    """
    Devuelve la moda de los píxeles válidos (>0) del raster
    dentro del polígono. Devuelve None si no hay píxeles.
    """
    try:
        with rasterio.open(tif_path) as src:
            # Comprobación rápida de bbox antes de enmascarar
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
# 4. Construcción del CSV
# ---------------------------------------------------------------------------

def generar_csv(gdf: gpd.GeoDataFrame, tif_map: dict, fecha_base: datetime, output: Path):
    n_dias = max(d for (_, d) in tif_map.keys()) + 1
    fechas = [(fecha_base + timedelta(days=d)).strftime("%d/%m") for d in range(n_dias)]

    output.parent.mkdir(parents=True, exist_ok=True)
    filas = []

    total = len(gdf)
    for i, row in gdf.iterrows():
        nombre = row["NAMEUNIT"]
        prov = row["provincia"] or ""
        geom = row.geometry

        vals = []
        for d in range(n_dias):
            v = extraer_riesgo(geom, tif_map.get(("p", d))) or \
                extraer_riesgo(geom, tif_map.get(("c", d))) or 1
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

    log.info(f"CSV generado: {output}  ({len(filas)} municipios)")
    return filas


# ---------------------------------------------------------------------------
# 5. Subida a Datawrapper
# ---------------------------------------------------------------------------

def subir_a_datawrapper(csv_path: Path):
    """
    Sube el CSV a Datawrapper y republica la visualización.
    Requiere DW_API_KEY y DW_CHART_ID en variables de entorno.
    Documentación: https://developer.datawrapper.de/reference/putchartsiddata
    """
    api_key  = os.getenv("DW_API_KEY")
    chart_id = os.getenv("DW_CHART_ID", "EqqfeFodYj")  # fallback al ID conocido

    if not api_key:
        log.info("DW_API_KEY no definido — se omite la subida a Datawrapper.")
        return

    headers = {"Authorization": f"Bearer {api_key}"}

    # Paso 1: subir los datos
    log.info(f"Subiendo datos a Datawrapper (chart {chart_id})...")
    with open(csv_path, encoding="utf-8-sig") as f:
        csv_text = f.read()

    r = requests.put(
        f"https://api.datawrapper.de/v3/charts/{chart_id}/data",
        headers={**headers, "Content-Type": "text/csv"},
        data=csv_text.encode("utf-8"),
        timeout=60,
    )
    if not r.ok:
        log.error(f"Error al subir datos: {r.status_code} {r.text}")
        return
    log.info("Datos subidos correctamente.")

    # Paso 2: republicar para que el embed se actualice
    r2 = requests.post(
        f"https://api.datawrapper.de/v3/charts/{chart_id}/publish",
        headers=headers,
        timeout=60,
    )
    if r2.ok:
        log.info("Visualización republicada correctamente.")
    else:
        log.error(f"Error al republicar: {r2.status_code} {r2.text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        tif_dir = descargar_zip(AEMET_URL, Path(tmpdir) / "tifs")
        fecha_base, tif_map = detectar_fecha_y_tifs(tif_dir)
        gdf = cargar_municipios(SHP_PEN, SHP_CAN)
        generar_csv(gdf, tif_map, fecha_base, OUTPUT_CSV)

    subir_a_datawrapper(OUTPUT_CSV)
    log.info("Proceso completado.")
