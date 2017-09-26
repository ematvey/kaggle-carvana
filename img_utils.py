import os

import cv2
import numpy as np

from conf import train_dataset
from dataset import (CARVANA_H, CARVANA_W, TRAIN_INDEX,
                     CarvanaDataset, carvana_pad_to_std)
from train_utils import var


def mask_to_bgr(mask, B, G, R):
    return cv2.cvtColor((mask / mask.max()).astype(np.uint8), cv2.COLOR_GRAY2BGR) * np.array([B, G, R], dtype=np.uint8)


def create_checkpoint_mask(img, mask, predicted_mask):
    p_mask = predicted_mask
    assert p_mask.shape[0] < p_mask.shape[1]
    if p_mask.shape == (CARVANA_H, CARVANA_W + 2):
        p_mask = p_mask[:, 1:-1]
    else:
        p_mask = cv2.resize(p_mask, (CARVANA_W, CARVANA_H),
                            interpolation=cv2.INTER_NEAREST)
    p_mask = (p_mask > 0.5).astype(np.uint8)
    true_mask = mask_to_bgr(mask, 0, 255, 0)
    p_mask = mask_to_bgr(p_mask, 0, 0, 255)
    w = cv2.addWeighted(img, 1.0, true_mask, 0.3, 0)
    w = cv2.addWeighted(w, 1.0, p_mask, 0.5, 0)
    return w


def draw_mask(out_fn, img_fn, mask_fn, predicted_mask):
    img = cv2.imread(img_fn)
    mask = np.load(mask_fn).astype(np.uint8)
    w = create_checkpoint_mask(img,mask,predicted_mask)
    cv2.imwrite(out_fn, w)


def output_some_images(model, dataset):
    os.makedirs('output', exist_ok=True)
    imgs = [0, 10, 20, 30]
    for i in imgs:
        fns = dataset.index.iloc[i]
        img_fn = fns['img']
        mask_fn = fns['mask']
        x = dataset[i][0]
        out = model(var(x).unsqueeze(0))
        out = out.data.cpu().numpy().squeeze()
        out_fn = os.path.join('output', img_fn.split('/')[-1])
        draw_mask(out_fn, img_fn, mask_fn, out)


def output_resized_mask():
    import pandas as pd
    os.makedirs('output/resize', exist_ok=True)
    imgs = [0, 10, 20, 30]
    df = pd.read_csv(TRAIN_INDEX)

    for i in imgs:
        fns = df.iloc[i]
        img_fn = fns['img']
        mask_fn = fns['mask']
        print('mask_fn', mask_fn)
        mask = carvana_pad_to_std(np.load(mask_fn))

        for downsample in [1.0, 1.5, 2.0, 4.0]:
            h = int(1280 / downsample)
            w = int(1920 / downsample)
            out_fn = os.path.join('output/resize/{}_{}x{}.png'.format(i, w, h))

            print(mask.shape)
            print((h, w))
            m = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_AREA)
            print(m.shape)

            draw_mask(out_fn, img_fn, mask_fn, m)
