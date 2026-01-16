import rasterio

def find_pixel_size(imagery_path:str)->int:
        with rasterio.open(imagery_path) as image:
            target_crs = rasterio.crs.CRS.from_string('EPSG:3857')
            # print(target_crs)
            img_crs = image.crs
            # print(img_crs)
            transform = image.transform
            width=image.width
            height=image.height
            left=transform[2]
            right = left+transform[0]*width
            bottom=transform[5]+transform[4]*height
            top=transform[5]
            pixel_size = rasterio.warp.calculate_default_transform(src_crs=img_crs,dst_crs=target_crs,width=width,height=height,left=left,right=right,bottom=bottom,top=top)[0][0]
            return pixel_size
        
