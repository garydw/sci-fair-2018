import openslide
import numpy as np
from openslide.deepzoom import DeepZoomGenerator
import math
import random
import cv2

global stain_ref 
global max_sat_ref
global max_color
stain_ref = [[0.50283219,  0.34453637],
             [0.7963168 ,  0.71543068],
             [0.33621323,  0.60782698]]
max_sat_ref = [[ 0.47047266],
               [ 0.58016781]]
max_color = [192.80795288085938, 167.06353378295898, 202.19671249389]

def normalize_staining(sample, beta=0.15, alpha=1, light_intensity=255):
    h, w, c = sample.shape
    pixel_ar = np.asarray(sample).reshape(-1, c)
    x = pixel_ar.astype(np.float64)
    od = -np.log10(x / light_intensity + 1e-8)
    od_thresh = od[np.all(od >= beta, 1), :]
    U, s, V = np.linalg.svd(od_thresh, full_matrices=False)
    top_eigvecs = V[0:2, :].T * -1
    proj = np.dot(od_thresh, top_eigvecs)
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)
    extreme_angles = np.array([[np.cos(min_angle), np.cos(max_angle)],
                               [np.sin(min_angle), np.sin(max_angle)]])
    stains = np.dot(top_eigvecs, extreme_angles)
    if stains[0, 0] < stains[0, 1]:
        stains[:, [0, 1]] = stains[:, [1, 0]]
    sats, _, _, _ = np.linalg.lstsq(stains, od.T)
    max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
    sats = sats / max_sat * max_sat_ref
    od_norm = np.dot(stain_ref, sats)
    x_norm = 10**(-od_norm) * light_intensity - 1e-8
    x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
    x_norm = x_norm.T.reshape(h, w, c)
    return x_norm

def find_refs(sample, beta=0.15, alpha=1, light_intensity=255):
    h, w, c = sample.shape
    pixel_ar = np.asarray(sample).reshape(-1, c)
    x = pixel_ar.astype(np.float64)
    od = -np.log10(x / light_intensity + 1e-8)
    od_thresh = od[np.all(od >= beta, 1), :]
    U, s, V = np.linalg.svd(od_thresh, full_matrices=False)
    top_eigvecs = V[0:2, :].T * -1
    proj = np.dot(od_thresh, top_eigvecs)
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)
    extreme_angles = np.array([[np.cos(min_angle), np.cos(max_angle)],
                               [np.sin(min_angle), np.sin(max_angle)]])
    stains = np.dot(top_eigvecs, extreme_angles)
    if stains[0, 0] < stains[0, 1]:
        stains[:, [0, 1]] = stains[:, [1, 0]]
    sats, _, _, _ = np.linalg.lstsq(stains, od.T)
    max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
    return (stains, max_sat)

def create_tile_generator(slide, tile_size, overlap):
    generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
    return generator

def get_zoom_level(slide, zoom, generator):
    highest_zoom_level = generator.level_count - 1
    try: 
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        offset = math.floor(mag / zoom / 2)
        level = highest_zoom_level - offset
    except (ValueError, KeyError) as e:
        level = highest_zoom_level
    return level

def pil2cv2(img):
    img = np.array(img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def select_tile(slide, zoom_level, h, w, tile_size):
    gen = create_tile_generator(slide, tile_size, 0)
    start_tile_r = h // tile_size #not index
    start_tile_c = w // tile_size #not index
    start_tile_h = h % tile_size
    start_tile_w = w % tile_size
    tiles = np.array([[pil2cv2(gen.get_tile(zoom_level, (start_tile_c - 1, start_tile_r - 1))), pil2cv2(gen.get_tile(zoom_level, (start_tile_c, start_tile_r - 1)))],
                      [pil2cv2(gen.get_tile(zoom_level, (start_tile_c - 1, start_tile_r))), pil2cv2(gen.get_tile(zoom_level, (start_tile_c, start_tile_r)))]])
    top_row = np.concatenate((tiles[0][0], tiles[0][1]), axis=1)
    bottom_row = np.concatenate((tiles[1][0], tiles[1][1]), axis=1)
    tile = np.concatenate((top_row, bottom_row), axis=0)
    section = tile[start_tile_r - 1:start_tile_r + 511, start_tile_c - 1:start_tile_c + 511,:]
    return section

def possible_coordinates(slide, zoom, num_imgs, tile_size):
    cv_im = np.array(slide.associated_images['thumbnail'].convert('RGB'))
    cv_im = cv2.cvtColor(cv_im, cv2.COLOR_RGB2BGR)
    avg_blur = cv2.blur(cv_im, (9, 9))
    gray_scale = cv2.cvtColor(avg_blur, cv2.COLOR_BGR2GRAY)
    ret, otsu = cv2.threshold(gray_scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(otsu, kernel, iterations=9) #h, w
    thumb_black = [(w, h) for h in range(len(dilated)) for w in range(len(dilated[0])) if dilated[h][w] == 0] #w, h
    gen = create_tile_generator(slide, tile_size, 0)
    level = get_zoom_level(slide, zoom, gen)
    level_w, level_h = gen.level_tiles[level]
    level_w *= tile_size
    level_h *= tile_size
    thumb_h, thumb_w = cv_im.shape[0:2]
    thumb_tile_h, thumb_tile_w = math.floor(tile_size / level_h * thumb_h), math.floor(tile_size / level_w * thumb_w)
    start_coords = []
    while len(start_coords) < num_imgs:
        rand = random.randint(0, len(thumb_black) - 1)
        start_coord = thumb_black[rand]
        if not (start_coord[0] + thumb_tile_w >= thumb_w or start_coord[1] + thumb_tile_h >= thumb_h):
            area = dilated[start_coord[1]:start_coord[1] + thumb_tile_h + 1, start_coord[0]:start_coord[0] + thumb_tile_w + 1]
            if not np.any(area > 0):
                if keep_image(select_tile(slide, level, math.floor(start_coord[1] / thumb_h * level_h), math.floor(start_coord[0] / thumb_w * level_w), tile_size)):
                    start_coords.append(start_coord)
    start_coords_conv = [(math.floor(y / thumb_h * level_h), math.floor(x / thumb_w * level_w)) for (x, y) in start_coords]
    return start_coords_conv

def create_images(file_name, zoom, num_imgs, tile_size):
    s = openslide.OpenSlide(file_name)
    coords = possible_coordinates(s, zoom, num_imgs, tile_size)
    gen = create_tile_generator(s, tile_size, 0)
    level = get_zoom_level(s, zoom, gen)
    tiles = []
    for coord in coords:
        tiles.append(normalize_staining(select_tile(s, level, coord[0], coord[1], tile_size)))
    return tiles

def keep_image(img):
    avg = [img[:, :, i].mean() for i in range(img.shape[-1])]
    avg_pix = np.array(avg).mean()
    if avg_pix >= 200:
        return False
    else:
        return True

def print_image(img):
    if isinstance(img, list):
        for i in range(len(img)):
            cv2.imshow('img{}'.format(i), img[i])
    else:
        cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, name):
    cv2.imwrite('{}.png'.format(name), image)