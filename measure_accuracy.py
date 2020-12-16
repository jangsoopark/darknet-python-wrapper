import pydarknet.darknet
import pydarknet.metrics

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

random.seed(999)


def scale_down(_image, iw, ih):
    _height, _width, _ = _image.shape
    while _height > ih or _width > iw:
        _image = cv2.resize(_image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        _height, _width, _ = _image.shape

    return _image


def draw(_image, detections, colors):
    for label, confidence, rect in detections:
        left, top, right, bottom = rect
        cv2.rectangle(_image, (left, top), (right, bottom), colors[label], 2)
        cv2.putText(
            _image, '{} [{:.2f}]'.format(label, float(confidence)),
            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2
        )
    return _image


# noinspection PyShadowingNames
def clip(bbox, sx, sy):
    rect = pydarknet.darknet.Darknet.bbox2rect(bbox)
    left, top, right, bottom = rect
    return (
        int(max(0, min(width, left * sx))),
        int(max(0, min(height, top * sy))),
        int(max(0, min(width, right * sx))),
        int(max(0, min(height, bottom * sy)))
    )


def init_results(classes):
    for c in classes:
        result_path = os.path.join('results', c)
        if os.path.exists(result_path):
            continue
        os.makedirs(result_path)


if __name__ == '__main__':
    model = pydarknet.darknet.Darknet()
    model.load_network(
        config='female-15k-tiny/cfg/yolov4-tiny.cfg',
        data='female-15k-tiny/data/female-15k.data',
        weights='female-15k-tiny/weights/yolov4-tiny.weights'
    )
    w, h = model.resolution
    init_results(model.class_names)

    with open('female-15k-tiny/data/valid.txt') as f:
        data_list = f.read().split('\n')
    class_colors = model.class_colors()
    class_names = model.class_names

    confusions = []

    precision_recall = {
        name: {'precision': [], 'recall': []}
        for name in class_names
    }

    for data_path in data_list:
        file_name = os.path.basename(data_path)

        image = cv2.imread(data_path)
        label = np.loadtxt(data_path.replace('jpg', 'txt')).reshape((-1, 5))

        image = scale_down(image, 1920, 1080)
        height, width, _ = image.shape

        dets = model.detect(image)
        c = pydarknet.metrics.confusion_metrics(label, dets, class_names, resolution=model.resolution)
        confusions.append(c)

    for c in confusions:
        for name, matrix in c.items():
            p = pydarknet.metrics.precision(matrix)
            r = pydarknet.metrics.recall(matrix)
            precision_recall[name]['precision'].append(p)
            precision_recall[name]['recall'].append(p)
        for name in c.keys():
            plt.plot(precision_recall[name]['precision'], precision_recall[name]['recall'])
            plt.show()
