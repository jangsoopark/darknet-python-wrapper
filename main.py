import pydarknet.darknet

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


def write_color_map(class_colors):
    radius = 50
    cmap = np.empty((radius, radius * len(class_colors), 3), dtype=np.uint8)
    i = 0
    for k, v in class_colors.items():
        for c in range(3):
            cmap[:, i * radius: (i + 1) * radius, c] = v[c]
        i += 1

    cv2.imwrite('results/color-map.png', cmap)


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
    write_color_map(class_colors)

    confusions = {
        name: {'tp': 0, 'fp': 0, 'fn': 0}
        for name in class_names
    }

    recall_precision = {
        name: []
        for name in class_names
    }

    for data_path in data_list:
        cname = os.path.basename(os.path.dirname(data_path))
        file_name = os.path.basename(data_path)

        image = cv2.imread(data_path)
        label = np.loadtxt(data_path.replace('jpg', 'txt')).reshape((-1, 5))

        image = scale_down(image, 1920, 1080)
        height, width, _ = image.shape

        dets = model.detect(image)

        sx, sy = width / w, height / h
        dets = [
            (d[0], d[1], clip(d[2], sx, sy))
            for d in dets
        ]

        image = draw(image, dets, class_colors)
        cv2.imshow('image', image)
        if cv2.waitKeyEx(0) & 0xff == ord('q'):
            break
        # cv2.imwrite(os.path.join('results', cname, file_name), image)
