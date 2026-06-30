#!/bin/bash

INPUT_DIR=$1
OUTPUT_GPKG=$2

echo "Input dir: $INPUT_DIR"
echo "Output gpkg: $OUTPUT_GPKG"
echo "Files found:"
ls "$INPUT_DIR"/*_AG_labels.tif

# rm -f "$OUTPUT_GPKG"

for tif in "$INPUT_DIR"/*_AG_labels.tif; do
    image_id=$(basename "$tif" _AG_labels.tif)
    
    # Check if layer already exists in gpkg
    if ogrinfo "$OUTPUT_GPKG" "$image_id" &>/dev/null 2>&1; then
        echo "Skipping $image_id — already exists"
        continue
    fi
    
    echo "Processing: $image_id"
    tmp_gpkg="/tmp/${image_id}_tmp.gpkg"
    gdal_polygonize.py -8 "$tif" -f GPKG "$tmp_gpkg" "labels"
    ogr2ogr -f GPKG -append -nln "$image_id" -where "DN != 7" "$OUTPUT_GPKG" "$tmp_gpkg" "labels"
    rm -f "$tmp_gpkg"
done

echo "Done!"