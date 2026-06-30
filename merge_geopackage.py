
#!/usr/bin/env python3
"""
merge_gpkg.py
 
Reads all layers from a source GeoPackage, fixes geometries, filters features
by area > 1000 m², and writes a single merged layer to a new GeoPackage.
 
Usage:
    python merge_gpkg.py input.gpkg output.gpkg [--area-threshold 1000] [--target-crs EPSG:32718]
"""
 
import argparse
import sys
import logging
from pathlib import Path
 
import fiona
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
from shapely.geometry import mapping
from tqdm import tqdm
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
 
def fix_geometry(geom):
    """Return a valid geometry, or None if it cannot be recovered."""
    if geom is None:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    # make_valid can return GeometryCollections; extract only polygonal parts
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        geom = polys[0] if len(polys) == 1 else polys[0].union(*polys[1:])
    return geom if geom.is_valid and not geom.is_empty else None
 
 
# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
 
def process_layer(src_path: str, layer_name: str, target_crs: str, area_threshold: float) -> gpd.GeoDataFrame | None:
    """
    Load one layer, fix geometries, reproject, filter by area.
    Returns a GeoDataFrame or None if no features survive.
    """
    try:
        gdf = gpd.read_file(src_path, layer=layer_name)
    except Exception as exc:
        log.warning(f"  Could not read layer '{layer_name}': {exc}")
        return None
 
    if gdf.empty:
        return None
 
    # Drop Z / M coordinates (can cause issues with some geometry ops)
    gdf.geometry = gdf.geometry.force_2d()
 
    # Fix invalid geometries
    gdf.geometry = gdf.geometry.apply(fix_geometry)
    gdf = gdf[gdf.geometry.notna()].copy()
 
    if gdf.empty:
        return None
 
    # Reproject to metric CRS for area computation
    if gdf.crs is None:
        log.warning(f"  Layer '{layer_name}' has no CRS — assuming {target_crs}")
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
 
    # Filter by area (already in metres² after reprojection)
    gdf = gdf[gdf.geometry.area > area_threshold].copy()
 
    if gdf.empty:
        return None
 
    # Tag with source layer name
    gdf["_source_layer"] = layer_name
 
    return gdf
 
 
def merge_gpkg(
    input_path: str,
    output_path: str,
    area_threshold: float = 1000.0,
    target_crs: str = "EPSG:32718",
    output_layer: str = "merged",
):
    input_path = str(input_path)
    output_path = str(output_path)
 
    # List all layers
    try:
        layers = fiona.listlayers(input_path)
    except Exception as exc:
        log.error(f"Cannot open '{input_path}': {exc}")
        sys.exit(1)
 
    log.info(f"Found {len(layers)} layers in {input_path}")
    log.info(f"Target CRS : {target_crs}")
    log.info(f"Area filter: > {area_threshold} m²")
 
    chunks: list[gpd.GeoDataFrame] = []
    skipped = 0
    total_in = 0
    total_out = 0
 
    for layer_name in tqdm(layers, desc="Processing layers", unit="layer"):
        if layer_name == "merged":
            print(f"skip {layer_name}")
            continue
        gdf = process_layer(input_path, layer_name, target_crs, area_threshold)
 
        if gdf is None or gdf.empty:
            skipped += 1
            continue
 
        total_out += len(gdf)
        chunks.append(gdf)
 
        # Flush to disk every 10 layers to keep RAM in check
        if len(chunks) >= 10:
            _flush(chunks, output_path, output_layer, target_crs)
            chunks = []
 
    # Final flush
    if chunks:
        _flush(chunks, output_path, output_layer, target_crs)
 
    log.info(f"Done. Layers processed: {len(layers) - skipped}  |  Layers skipped: {skipped}")
    log.info(f"Features written: {total_out}")
    log.info(f"Output: {output_path}  (layer='{output_layer}')")
 
 
def _flush(chunks: list[gpd.GeoDataFrame], output_path: str, layer_name: str, crs: str):
    """Concatenate chunk list and append to the output GeoPackage."""
    merged = pd.concat(chunks, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs=crs)
 
    # Determine write mode: 'w' for first write, 'a' to append
    mode = "a" if Path(output_path).exists() else "w"
 
    merged.to_file(
        output_path,
        layer=layer_name,
        driver="GPKG",
        mode=mode,
        # append=True is implied by mode='a' in recent geopandas
    )
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge GeoPackage layers → single filtered layer."
    )
    parser.add_argument("input",  help="Path to input GeoPackage")
    parser.add_argument("output", help="Path to output GeoPackage")
    parser.add_argument(
        "--area-threshold", type=float, default=1000.0,
        help="Minimum feature area in m² (default: 1000)",
    )
    parser.add_argument(
        "--target-crs", default="EPSG:3857",
        help="Metric CRS for reprojection / area calculation (default: EPSG:32718 — UTM 18S)",
    )
    parser.add_argument(
        "--output-layer", default="merged",
        help="Name of the output layer (default: 'merged')",
    )
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    merge_gpkg(
        input_path=args.input,
        output_path=args.output,
        area_threshold=args.area_threshold,
        target_crs=args.target_crs,
        output_layer=args.output_layer,
    )
 
