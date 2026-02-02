import rasterio

# def find_pixel_size(imagery_path:str)->int:
#         with rasterio.open(imagery_path) as image:
#             target_crs = rasterio.crs.CRS.from_string('EPSG:3857')
#             # print(target_crs)
#             img_crs = image.crs
#             # print(img_crs)
#             transform = image.transform
#             width=image.width
#             height=image.height
#             left=transform[2]
#             right = left+transform[0]*width
#             bottom=transform[5]+transform[4]*height
#             top=transform[5]
#             pixel_size = rasterio.warp.calculate_default_transform(src_crs=img_crs,dst_crs=target_crs,width=width,height=height,left=left,right=right,bottom=bottom,top=top)[0][0]
#             return pixel_size
        
import rasterio
from rasterio.warp import calculate_default_transform
from pyproj import CRS

def find_pixel_size(imagery_path: str) -> float:
    with rasterio.open(imagery_path) as src:
        # 1. Get the center of the image in its native CRS (4326)
        lon = src.lnglat()[0]
        lat = src.lnglat()[1]
        
        # 2. Determine the correct UTM EPSG code automatically
        # UTM zones are 6 degrees wide; 32700 is the prefix for Southern Hemisphere
        zone = int((lon + 180) / 6) + 1
        utm_crs = f"+proj=utm +zone={zone} +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        
        # 3. Calculate transform to that specific UTM zone
        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)
        
        # transform[0] is the pixel width in meters
        return abs(transform[0])