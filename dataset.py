import os
import random
import re
import warnings
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import tqdm
from torch.utils.data import Dataset

CARVANA_W = 1918
CARVANA_H = 1280

TRAIN_IMAGE_PATH = 'input/train'
TEST_IMAGE_PATH = 'input/test'
MASKS_DECODED_PATH = 'input/train_masks_decoded'

TRAIN_INDEX = 'input/train_index.csv'
DEV_INDEX = 'input/dev_index.csv'
CHANNELS_MEAN = 'input/chan_mean.npy'
CHANNELS_STD = 'input/chan_std.npy'


def run_length_decode(rel, fill_value=255):
    mask = np.zeros((CARVANA_H * CARVANA_W), np.uint8)
    rel = np.array([int(s) for s in rel.split(' ')]).reshape(-1, 2)
    for r in rel:
        start = r[0]
        end = start + r[1]
        mask[start:end] = fill_value
    mask = mask.reshape(CARVANA_H, CARVANA_W)
    return mask


def run_length_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def prepare_train_masks(masks_df):
    os.makedirs(MASKS_DECODED_PATH, exist_ok=True)
    print('preparing masks...')
    for _, row in tqdm.tqdm(masks_df.iterrows()):
        img = row['img']
        rle_mask = row['rle_mask']
        np.save(os.path.join(MASKS_DECODED_PATH, img + '.mask'),
                run_length_decode(rle_mask))


def check_masks(masks_df):
    # check masks
    train_files = os.listdir(TRAIN_IMAGE_PATH)
    masks = set(masks_df.img.values)
    for f in train_files:
        if not f.endswith('.jpg'):
            warnings.warn('not checking {}'.format(f))
            continue
        if not f in masks:
            warnings.warn('file without mask: {}'.format(f))
        else:
            masks.remove(f)
    if masks:
        warnings.warn('masks without files: {}'.format(masks))


def prepare_splits():
    # train-dev split
    train_files = os.listdir(TRAIN_IMAGE_PATH)
    regex = re.compile('([a-zA-Z0-9]+)_([0-9]{2}).jpg')
    cars_index = defaultdict(list)
    for f in train_files:
        car, _ = regex.match(f).groups()
        cars_index[car].append(f)
    cars = list(cars_index.keys())
    random.shuffle(cars)

    boundary = int(len(cars) * .8)
    train_cars = cars[:boundary]
    dev_cars = cars[boundary:]

    def make_index(cars_list):
        df = []
        for car in cars_list:
            for car_f in cars_index[car]:
                df.append({'img': os.path.join(TRAIN_IMAGE_PATH, car_f),
                           'mask': os.path.join(MASKS_DECODED_PATH, car_f + '.mask.npy')})
        return pd.DataFrame(df)

    make_index(train_cars).to_csv(TRAIN_INDEX, index=False)
    make_index(dev_cars).to_csv(DEV_INDEX, index=False)
    print('train-test split indices are prepared')


def calculate_mean_image(max_img=100):
    print('calculating mean image...')
    img_mean_sum = np.zeros(3, np.float64)
    img_var_sum = np.zeros(3, np.float64)

    train_files = os.listdir(TRAIN_IMAGE_PATH)
    random.shuffle(train_files)

    for i, f in tqdm.tqdm(enumerate(train_files)):
        channels = cv2.imread(os.path.join(
            TRAIN_IMAGE_PATH, f)).reshape((-1, 3))
        img_mean_sum += channels.mean(axis=0)
        img_var_sum += channels.var(axis=0)
        if i >= max_img:
            break

    chan_mean = img_mean_sum / (i + 1)
    chan_std = (img_var_sum / (i + 1)) ** 0.5

    print('channels mean:', chan_mean)
    print('channels std:', chan_std)
    np.save(CHANNELS_MEAN, chan_mean)
    np.save(CHANNELS_STD, chan_std)


def prepare_all(masks_input_csv_path='input/train_masks.csv'):
    masks_df = pd.read_csv(masks_input_csv_path)
    if not os.path.exists(MASKS_DECODED_PATH):
        prepare_train_masks(masks_df)
        check_masks(masks_df)
    if not os.path.exists(TRAIN_INDEX):
        prepare_splits()
    if not os.path.exists(CHANNELS_MEAN):
        calculate_mean_image(max_img=300)


def carvana_pad_to_std(img):
    assert img.shape[0] == 1280 and img.shape[1] == 1918
    # H, W
    if len(img.shape) == 2:
        return np.pad(img, [[0, 0], [1, 1]], mode='edge')
    # H, W, C
    if len(img.shape) == 3:
        return np.pad(img, [[0, 0], [1, 1], [0, 0]], mode='edge')
    # ???
    raise ValueError()


def carvana_img_downsample(img, H, W):
    assert img.shape[0] < img.shape[1]
    if H == CARVANA_H and W == CARVANA_W:
        carvana_pad_to_std(img)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def carvana_img_restore(img):
    assert img.shape[0] < img.shape[1]
    if img.shape == (CARVANA_H, CARVANA_W + 2):
        return img[:, 1:-1]
    return cv2.resize(img, (CARVANA_W, CARVANA_H), interpolation=cv2.INTER_LINEAR)


class CarvanaLoader(object):
    def __init__(self, H, W):
        """shape is (H, W)"""
        self.H, self.W = H, W
        self.channels_mean = np.load(CHANNELS_MEAN).astype(np.float32)
        self.channels_std = np.load(CHANNELS_STD).astype(np.float32)

    def get_image(self, img_file):
        img = cv2.imread(img_file).astype(np.float32)
        img = carvana_img_downsample(img, self.H, self.W)
        img = (img.astype(np.float32) - self.channels_mean) / self.channels_std
        img = img.transpose(2, 0, 1)
        return img

    def get_mask(self, mask_file):
        mask = np.load(mask_file).astype(np.float32) / 255
        mask = carvana_img_downsample(mask, self.H, self.W)
        return mask


class CarvanaDataset(Dataset, CarvanaLoader):
    def __init__(self, index_file, shape):
        """shape is (H, W)"""
        H, W = shape
        super().__init__(H, W)
        self.index_file = index_file
        self.index = pd.read_csv(index_file)

    def __getitem__(self, index):
        img_file, mask_file = self.index.values[index]
        return self.get_image(img_file), self.get_mask(mask_file)

    def __len__(self):
        return self.index.shape[0]


class CarvanaTestDataset(Dataset, CarvanaLoader):
    def __init__(self, shape):
        """shape is (H, W)"""
        H, W = shape
        super().__init__(H, W)
        self.index = [f for f in os.listdir(
            TEST_IMAGE_PATH) if f.endswith('.jpg')]

    def __getitem__(self, index):
        img_fn = self.index[index]
        img_file = os.path.join(TEST_IMAGE_PATH, img_fn)
        return img_fn, self.get_image(img_file)

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    prepare_all()
