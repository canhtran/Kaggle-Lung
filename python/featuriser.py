img_rows = 512
img_cols = 512
smooth = 1.

import os
from multiprocessing import Pool
import pickle
import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation

files = [i+'.npy' for i in pd.read_csv('./missing.csv')['id']]

def get_extractor():
    model = mx.model.FeedForward.load('./resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_data_id(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0
    # f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

        # if cnt < 20:
        #     plots[cnt // 5, cnt % 5].axis('off')
        #     plots[cnt // 5, cnt % 5].imshow(np.swapaxes(tmp, 0, 2))
        # cnt += 1

    # plt.show()
    batch = np.array(batch)
    return batch
    
net = get_extractor()
def calc_features(id_):
    batch = np.mean(np.load('./stage1.1/%s' % str(files[id_])), axis=0)
    batch = np.array([batch])
    print("{} is Found ".format(str(id_)))

    feats = net.predict(batch)
    print(feats.shape)
    np.save('./stage1.1_features/{}'.format(files[id_]), feats)

if __name__ == '__main__':
    p = Pool(processes = 15)
p.map(calc_features, range(2, len(files)))
