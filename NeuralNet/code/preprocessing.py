import cv2
import numpy as np
from scipy import ndimage
import random
import pyodbc
import os

global pixel_avg
pixel_avg = np.array([195.97876791905875, 165.11959514039935, 197.0193379835649])

def read_files(directory, objective_num):
    files = []
    for nm in os.listdir(directory):
        if nm[-3:] == 'png' and 'objective{}'.format(objective_num) in nm:
            files.append(nm)
    return files

def read_images(files, directory):
    images = [cv2.imread(directory + '\\' + file) for file in files]
    return np.array(images)

def horizontal_flip(images):
    flipped = [cv2.flip(img, 0) for img in images]
    return flipped

def vertical_flip(images):
    flipped = [cv2.flip(img, 1) for img in images]
    return flipped

def rotations(images):
    im_shape = np.array(images).shape
    rotated = np.array([np.array([ndimage.rotate(img, (x + 1) * 90) for x in range(3)]) for img in images])
    flattened = rotated.reshape(im_shape[0] * 3, im_shape[1], im_shape[2], im_shape[3])
    return flattened

def noise(images):
    rand = np.random.randint(0, 50, tuple(np.array(images).shape))
    noised = images + rand
    return noised            
    
def augment_data(training, labels, ids):
    horz = horizontal_flip(training)
    vert = vertical_flip(training)
    noised = noise(training)
    rot = rotations(training)
    new_labels, lengthened_ids = np.concatenate((np.tile(labels, 3), np.repeat(labels, 3))), np.concatenate((np.tile(ids, 3), np.repeat(ids, 3)))
    new_ids = []
    for i in range(len(lengthened_ids)):
        if i < len(ids):
            new_ids.append('horz' + lengthened_ids[i])
        elif i < 2 * len(ids):
            new_ids.append('vert' + lengthened_ids[i])
        elif i < 3 * len(ids):
            new_ids.append('nois' + lengthened_ids[i])
        elif (i % (len(ids) * 3)) % 3 == 0:
            new_ids.append('rot0' + lengthened_ids[i])
        elif (i % (len(ids) * 3)) % 3 == 1:
            new_ids.append('rot1' + lengthened_ids[i])
        elif (i % (len(ids) * 3)) % 3 == 2:
            new_ids.append('rot2' + lengthened_ids[i])
    new_training = np.concatenate((horz, vert, noised, rot))
    return new_training, new_labels, new_ids

def center_images(images):
    new_images = images - pixel_avg
    return new_images

def get_cursor():
    server = r'localhost\SQLEXPRESS'
    database = 'Patients'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
    cursor = cnxn.cursor()
    cnxn.autocommit = True
    return cursor

def get_labels(files, cursor):
    total_labels = []
    for patient in files:
        cursor.execute("select EntityID, concat(GleasonGradePrimary, '+', GleasonGradeSecondary) from Patient where EntityID = '{}'".format(patient[0:12]))
        total_labels.append(cursor.fetchone())
    ids, labels = [point[0] for point in total_labels], [point[1] for point in total_labels]
    return labels, ids

def inputs_labels_ids(directory, objective_num): 
    files = read_files(directory, objective_num)
    images = read_images(files, directory)
    cursor = get_cursor()
    labels, ids = get_labels(files, cursor)
    return images, labels, ids

def get_avg_pix(img):
    avg = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return avg

def get_total_avg(images):
    avgs = np.array([get_avg_pix(img) for img in images])
    total_avg = [avgs[:, i].mean() for i in range(len(avgs[0]))]
    return total_avg

def encode_labels(labels):
    numerical = [0 if label == '3+4' else 1 for label in labels] # 3+4 = 0 4+3 = 1
    return numerical

def split_train_test(data, ids, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices], ids[train_indices], ids[test_indices]

def images_and_labels(batch_num, batch_size, ids, data_num, train=True):
    directory = r'C:\Users\matthew\Documents\ScienceFair2017-2018\ManifestDownloads\data{}\train'.format(data_num) + '\\' if train else r'C:\Users\matthew\Documents\ScienceFair2017-2018\ManifestDownloads\data{}\test'.format(data_num) + '\\'
    images = []
    labels = []
    image_names = [name for name in os.listdir(directory) if name[-3:] != 'txt']
    entities = [name[-27:-4] if name[-28:-27] == '_' else name[-31:-4] for name in image_names]
    file = open(directory + 'labels.txt', 'r')
    labels_list = file.readlines()
    for i in range(len(ids)):
        labels.append(labels_list[batch_num * batch_size + i][:-1])
        images.append(cv2.imread(directory + image_names[entities.index(ids[i])]))
    return images, labels

def write_data(training, labels, ids, data_num, train='train'):
    directory = r'C:\Users\matthew\Documents\ScienceFair2017-2018\ManifestDownloads\data{}'.format(data_num) + '\\' + train + '\\'
    file = open(directory + 'labels.txt', 'a+')
    for label in labels:
        file.write(str(label) + '\n')
    file.close()
    file = open(directory + 'ids.txt', 'a+')
    lengthened_ids = []
    for i in range(len(training)):
        num = str(round(random.random() * 10e15))[0:10]
        cv2.imwrite(directory + 'img_{}_{}.png'.format(ids[i], num), training[i])
        file.write(str(ids[i]) + '_' + num + '\n')
        lengthened_ids.append(str(ids[i]) + '_' + num)
    file.close()
    return lengthened_ids

def prepare_data(directory, test_ratio, objective_num, data_num): 
    random.seed(42)
    total_images, total_labels, total_ids = inputs_labels_ids(directory, objective_num)
    total_data = [(img, lab) for img, lab in zip(total_images, total_labels)]
    train_data, test_data, train_ids, test_ids = split_train_test(np.array(total_data), np.array(total_ids), test_ratio)
    pre_train_images, pre_train_labels = np.array([point[0] for point in train_data]), np.array([point[1] for point in train_data])
    test_features, test_labels = np.array([point[0] for point in test_data]), np.array([point[1] for point in test_data])
    new_train_ids = write_data(pre_train_images, pre_train_labels, train_ids, data_num)
    new_test_ids = write_data(test_features, test_labels, test_ids, data_num, train='test')
    #del total_images, total_labels, total_data, train_data, test_data, pre_train_images, pre_train_labels, test_features, test_labels
    #for i in range(len(new_train_ids) // 20):
    #    pre_img, pre_label = images_and_labels(i, 20, new_train_ids[20 * i: 20 * i + 20], data_num)
    #    post_img, post_label, post_ids = augment_data(pre_img, pre_label, train_ids[20 * i: 20 * i + 20])
    #    write_data(post_img, post_label, post_ids, data_num)

def data(batch_num, batch_size, data_num, train_or_test='train'):
    data_directory = r'C:\Users\matthew\Documents\ScienceFair2017-2018\ManifestDownloads\data{}'.format(data_num) + '\\' + train_or_test + '\\ids.txt'
    train = True if train_or_test == 'train' else False
    f = open(data_directory, 'r')
    non_delimit = [line[:-1] for line in f.readlines()]
    f.close()
    ids = non_delimit[batch_num * batch_size: batch_num * batch_size + batch_size]
    images, labels = images_and_labels(batch_num, batch_size, ids, data_num, train)
    images = np.array(images, dtype=np.float32)
    images /= np.array([255, 255, 255]).astype(np.float32)
    #images -= pixel_avg.astype(np.float32)
    return images, encode_labels(labels)

prepare_data(r'C:\Users\matthew\Documents\ScienceFair2017-2018\ManifestDownloads\SVSManifest', 0.2, 30, 1)