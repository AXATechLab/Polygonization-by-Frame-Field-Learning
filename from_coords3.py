import utm
import mpu
import os
import io
import cv2
import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import mercantile
import requests
import shutil
from PIL import Image
from scipy import ndimage
from shapely import geometry
#from rdp import rdp
from network import *

def segment_buildings(model_weight_path, bounding_box, mapbox_api_key, parcel_polygon=None, cpu=True, building_size_min=30.0, fire_dist=5.0):
    """
       Inpout :
       model_weight_path (string) : Contains the path towards the file containing the weights of the segmentation NN
       bounding_box ((float, float), (float, float)): contains the (lat, long) coordinates of the region of interest, the first one determines the top left corner, the second one determines the bottom right corner
       mapbox_api_key (string): The mapbox_api_key linked with the mapbox account, to be able to make requests
       parcel_polygon list[(float, float)]: Contains a list of latitude/longitude coordinates of the parcel of interest
       building_size_min (float) : minimum area in squared meter of a building in order to ba taken into account
       fire_dist (float) : maximum distance to consider a pair of buildings to be linked
       
       Output :
       image (bytes array): Satellite image delimited by the bounding box
       pred (bytes array): Pixelwise binary prediction (building or not building) on the image
       buildings (list[dict]): Contains various information on each building, the xy and latitude/longitude coordinates of the contour of the building, the area and the connected component the building is in
       dists (numpy array): Contains the distance between each pair of buildings
       nb_comp (int): contains the number of connected component (buildings are linked if they are less than 5 meters apart)
       parcel_polygon_xy : the pixel coordinates of the parcel
    """
    
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], strides=(2, 2, 2, 2, 2), inter_features=True)
    model = DLinkNet2(10, 2, resnet)
    if cpu:
        model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_weight_path))
        model.cuda()
    top_left = bounding_box[0]
    bottom_right = bounding_box[1]
    
    latitude_magnitude = np.abs(top_left[0] / 2 + bottom_right[0] / 2)
    if latitude_magnitude <= 30.0:
        z = 18
    elif latitude_magnitude <= 50.0:
        z = 17
    elif latitude_magnitude <= 70.0:
        z = 16
    else:
        z = 15

    top_left_tile = mercantile.tile(top_left[1], top_left[0], z)
    bottom_right_tile = mercantile.tile(bottom_right[1], bottom_right[0], z)
    x_tile_range =[top_left_tile.x,bottom_right_tile.x]
    y_tile_range = [top_left_tile.y, bottom_right_tile.y]
    
    big_image = np.zeros(((y_tile_range[1] - y_tile_range[0] + 1) * 512, \
                          (x_tile_range[1] - x_tile_range[0] + 1) * 512, 3))
    
    westernmost = 200.0
    northernmost = -200.0
    southernmost = 200.0
    easternmost = -200.0
    
    for i, x in enumerate(range(x_tile_range[0], x_tile_range[1]+1)):
        for j, y in enumerate(range(y_tile_range[0], y_tile_range[1]+1)):
            west, south, east, north = mercantile.bounds(x, y, z)
            if west <= westernmost:
                westernmost = west
            if south <= southernmost:
                southernmost = south
            if east >= easternmost:
                easternmost = east
            if north >= northernmost:
                northernmost = north
                
            url = 'https://api.mapbox.com/v4/mapbox.satellite/'+ str(z)+'/'+str(x)+'/'+str(y)+'@2x.pngraw?access_token=' + mapbox_api_key
            r = requests.get(url, stream=True, verify=True)
            temp_im = np.array(Image.open(io.BytesIO(r.content)))[:,:,:3]
            big_image[j*512:(j+1)*512, i*512:(i+1)*512, :] = temp_im
            
    
    model.eval()
    img = process_image(big_image, cpu).unsqueeze(0)

    patch_tile_x = x_tile_range[1] - x_tile_range[0] + 1
    patch_tile_y = y_tile_range[1] - y_tile_range[0] + 1
    while patch_tile_x * patch_tile_y >= 10:
        if patch_tile_x >= patch_tile_y:
            patch_tile_x -= 1
        else:
            patch_tile_y -= 1
    patch_x = patch_tile_x * 512
    patch_y = patch_tile_y * 512

    if x_tile_range[1] - x_tile_range[0] + 1 == patch_tile_x:
        stride_x = 1
        x_range = 1
    elif x_tile_range[1] - x_tile_range[0] + 1 - patch_tile_x <= patch_tile_x / 2:
        stride_x = (x_tile_range[1] - x_tile_range[0] + 1 - patch_tile_x) * 512
        x_range = 2
    else:
        diff = 1e+10
        for stride in range(1, (x_tile_range[1] - x_tile_range[0] + 1 - patch_tile_x) + 1):
            new_diff = np.abs(patch_tile_x / 2 - stride)
            if (x_tile_range[1] - x_tile_range[0] + 1 - patch_tile_x) % stride == 0 and new_diff <= diff:
                stride_x = stride * 512
                diff = new_diff
        x_range = ((x_tile_range[1] - x_tile_range[0] + 1) * 512 - patch_x) // stride_x + 1

    if y_tile_range[1] - y_tile_range[0] + 1 == patch_tile_y:
        stride_y = 1
        y_range = 1
    elif y_tile_range[1] - y_tile_range[0] + 1 - patch_tile_y <= patch_tile_y / 2:
        stride_y = (y_tile_range[1] - y_tile_range[0] + 1 - patch_tile_y) * 512
        y_range = 2
    else:
        diff = 1e+10
        for stride in range(1, (y_tile_range[1] - y_tile_range[0] + 1 - patch_tile_y) + 1):
            new_diff = np.abs(patch_tile_x / 2 - stride)
            if (y_tile_range[1] - y_tile_range[0] + 1 - patch_tile_y) % stride == 0 and new_diff <= diff:
                stride_y = stride * 512
                diff = new_diff
        y_range = ((y_tile_range[1] - y_tile_range[0] + 1) * 512 - patch_y) // stride_y + 1

    results = []
    for i in range(x_range):
        temp_results = []
        for j in range(y_range):
            pred = torch.nn.Softmax(0)(predict_batch(i, j, x_range, img, model, patch_x, patch_y, \
                                                     stride_x, stride_y, cpu)[0,...])
            pred = pred[1,:,:].detach().cpu().numpy()
            temp_results.append(pred)
        results.append(temp_results)
        
    degree = 15

    tri_x = scipy.signal.triang(patch_x)
    tri_y = scipy.signal.triang(patch_y)

    filter_x = tri_x
    filter_y = tri_y
    for i in range(degree):
        filter_x = np.convolve(filter_x, tri_x, mode='same')
        filter_y = np.convolve(filter_y, tri_y, mode='same')

    filter_x = (filter_x + filter_x[::-1]) / 2
    filter_x = filter_x / max(filter_x)
    filter_y = (filter_y + filter_y[::-1]) / 2
    filter_y = filter_y / max(filter_y)

    total_filter = np.expand_dims(filter_y, axis=1) @ np.expand_dims(filter_x, axis=0)

    big_pred = np.zeros((big_image.shape[0], big_image.shape[1]))
    filter_pred = np.zeros((big_image.shape[0], big_image.shape[1]))
    for i in range(x_range):
        for j in range(y_range):
            batch = results[i][j]
            big_pred[j * stride_y:j * stride_y + patch_y, i * stride_x:i * stride_x + patch_x] += batch * total_filter
            filter_pred[j*stride_y:j * stride_y + patch_y, i * stride_x:i * stride_x + patch_x] += total_filter
            
    big_pred = big_pred / (filter_pred + 1e-8)
    big_pred = np.where(big_pred > 0.5, 1, 0)
    
    eps = 0.025
    relative_pos_tl = [max((northernmost - top_left[0]) / (northernmost - southernmost) - eps, 0), \
                       max((top_left[1] - westernmost) / (easternmost - westernmost) - eps, 0)]
    
    relative_pos_br = [min((northernmost - bottom_right[0]) / (northernmost - southernmost) + eps, 1), \
                       min((bottom_right[1] - westernmost) / (easternmost - westernmost) + eps, 1)]
    
    pos_tl = [int(np.round(relative_pos_tl[0] * big_image.shape[0])), \
              int(np.round(relative_pos_tl[1] * big_image.shape[1]))]
    
    pos_br = [int(np.round(relative_pos_br[0] * big_image.shape[0])), \
              int(np.round(relative_pos_br[1] * big_image.shape[1]))]
    
    image = big_image[pos_tl[0]:pos_br[0], pos_tl[1]:pos_br[1],:].astype(np.uint8)
    
    pred = big_pred[pos_tl[0]:pos_br[0], pos_tl[1]:pos_br[1]].astype(np.uint8)
    
    parcel_polygon_xy = []
    if parcel_polygon is not None:
        for pa_poly in parcel_polygon:
            pa_poly_xy = []
            for coord in pa_poly:
                relative_coord = [(northernmost - coord[0]) / (northernmost - southernmost), \
                                  (coord[1] - westernmost) / (easternmost - westernmost)]
                xy_coord = [int(np.round(relative_coord[1] * big_image.shape[1]) - pos_tl[1]), \
                            int(np.round(relative_coord[0] * big_image.shape[0]) - pos_tl[0])]
                pa_poly_xy.append(xy_coord)
            pa_poly_xy.append(pa_poly_xy[0])
            parcel_polygon_xy.append(pa_poly_xy)
            
    
    if parcel_polygon is not None:
        binary_parcel = np.zeros(big_pred.shape)
        for pa_poly_xy in parcel_polygon_xy:
            cv2.fillPoly(binary_parcel[pos_tl[0]:pos_br[0], pos_tl[1]:pos_br[1]], [np.array(pa_poly_xy)], 1)
    
    if parcel_polygon is not None:
        big_pred = big_pred * binary_parcel
    big_contours = cv2.findContours(big_pred.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_contours = big_contours[0]
    big_contours = [rdp(contour.squeeze(1), epsilon=2.0) for contour in big_contours]
    i = 1
    buildings = []
    for contour in big_contours:
        i += 1
        keep = False
        info = {}
        contour_xy = []
        contour_lat_long = []
        
        for coord in contour:
            contour_xy.append((np.clip(coord[0] - pos_tl[1], 0, image.shape[1] - 1), np.clip(coord[1] - pos_tl[0], 0, \
                                                                                         image.shape[0] - 1)))
            long = westernmost + coord[0] * (easternmost - westernmost) / big_image.shape[1]
            lat = northernmost - coord[1] * (northernmost - southernmost) / big_image.shape[0]
            contour_lat_long.append((lat, long))
        contour_xy.append(contour_xy[0])
        contour_lat_long.append(contour_lat_long[0])
        info['xy'] = contour_xy
        info['lat_long'] = contour_lat_long
        if len(contour_lat_long) <= 3:
            info['area'] = 0.0
        else:
            info['area'] = get_area(contour_lat_long)
        if info['area'] > 0.0:
            buildings.append(info)
            
    buildings = sorted(buildings, key=lambda k: -k['area'])
    i = 1
    for building in buildings:
        building['index'] = i
        i += 1
        
    image_bytes = io.BytesIO()
    np.save(image_bytes, image, allow_pickle=True)
    image = image_bytes.getvalue()
    
    buildings = geojson.dumps(buildings)
    buildings = geojson.loads(buildings)
    
    return image, buildings, parcel_polygon_xy

def process_image(im, cpu):
    
    if not cpu:
        X = torch.tensor(im.copy()).permute(2, 0, 1).float()
        X = X.cuda()
    else:
        X = torch.tensor(im.copy(), device=torch.device('cpu')).permute(2, 0, 1).float()
    X[0,:,:] = (X[0,:,:] - X[0,:,:].mean()) / X[0,:,:].std()
    X[1,:,:] = (X[1,:,:] - X[1,:,:].mean()) / X[1,:,:].std()
    X[2,:,:] = (X[2,:,:] - X[2,:,:].mean()) / X[2,:,:].std()
    
    return X

def get_area(coords):
    coords = [(utm.from_latlon(coord[0], coord[1])[0], utm.from_latlon(coord[0], coord[1])[1]) for coord in coords]
    return geometry.Polygon(coords).area
            
def get_dist(buildings, image, big_image, westernmost, northernmost, southernmost, easternmost, pos_tl):

    dist_matrix = np.zeros((len(buildings), len(buildings)))
    for i in range(len(buildings)):
        for j in range(i + 1, len(buildings)):
            mask1 = np.zeros((image.shape[0], image.shape[1]))
            mask2 = np.zeros((image.shape[0], image.shape[1]))
            cv2.fillPoly(mask1, [np.array(buildings[i]['xy'])], 1)
            cv2.fillPoly(mask2, [np.array(buildings[j]['xy'])], 1)
            dist1 = ndimage.distance_transform_edt(1 - mask1)
            dist2 = ndimage.distance_transform_edt(1 - mask2)

            closest_pixel1 = np.unravel_index(np.where(dist2 * mask1 == 0, 10**12, dist2 * mask1).argmin(), \
                                              mask1.shape)
            closest_pixel2 = np.unravel_index(np.where(dist1 * mask2 == 0, 10**12, dist1 * mask2).argmin(), \
                                              mask1.shape)

            long1 = westernmost + (closest_pixel1[0] + pos_tl[0]) * (easternmost - westernmost) / big_image.shape[0]
            lat1 = northernmost - (closest_pixel1[1] + pos_tl[1]) * (northernmost - southernmost) / big_image.shape[1]

            long2 = westernmost + (closest_pixel2[0] + pos_tl[0]) * (easternmost - westernmost) / big_image.shape[0]
            lat2 = northernmost - (closest_pixel2[1] + pos_tl[1]) * (northernmost - southernmost) / big_image.shape[1]

            dist = mpu.haversine_distance((lat1, long1), (lat2, long2)) * 1000

            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist
            
    return dist_matrix

def predict_batch(i, j, x_range, big_image, model, patch_x, patch_y, stride_x, stride_y, cpu):
    
    batch = big_image[:, :, j * stride_y:j * stride_y + patch_y, i * stride_x:i * stride_x + patch_x]
    if not cpu:
        batch = batch.cuda()
    _, pred = model(batch)
    return pred
