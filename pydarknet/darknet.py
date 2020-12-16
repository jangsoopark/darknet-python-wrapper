from . import _load_lib
from . import utils

import ctypes
import random
import cv2


class Darknet(object):

    def __init__(self):
        self.lib = _load_lib.load()

        self.network = None
        self.class_names = None
        self.image = None
        self.resolution = ()

        # initialize library
        self.lib.network_width.argtypes = [ctypes.c_void_p]
        self.lib.network_width.restype = ctypes.c_int
        self.lib.network_height.argtypes = [ctypes.c_void_p]
        self.lib.network_height.restype = ctypes.c_int

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [utils.IMAGE, ctypes.c_char_p]

        self.predict = self.lib.network_predict_ptr
        self.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.predict.restype = ctypes.POINTER(ctypes.c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [ctypes.c_int]

        self.init_cpu = self.lib.init_cpu

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.make_image.restype = utils.IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.get_network_boxes.restype = ctypes.POINTER(utils.DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [ctypes.c_void_p]
        self.make_network_boxes.restype = ctypes.POINTER(utils.DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [ctypes.POINTER(utils.DETECTION), ctypes.c_int]

        self.free_batch_detections = self.lib.free_batch_detections
        self.free_batch_detections.argtypes = [ctypes.POINTER(utils.DETNUMPAIR), ctypes.c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

        self.network_predict = self.lib.network_predict_ptr
        self.network_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [ctypes.c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.load_net.restype = ctypes.c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.load_net_custom.restype = ctypes.c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [ctypes.POINTER(utils.DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [ctypes.POINTER(utils.DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [utils.IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [utils.IMAGE, ctypes.c_int, ctypes.c_int]
        self.letterbox_image.restype = utils.IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [ctypes.c_char_p]
        self.lib.get_metadata.restype = utils.METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.load_image.restype = utils.IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [utils.IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [ctypes.c_void_p, utils.IMAGE]
        self.predict_image.restype = ctypes.POINTER(ctypes.c_float)

        self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [ctypes.c_void_p, utils.IMAGE]
        self.predict_image_letterbox.restype = ctypes.POINTER(ctypes.c_float)

        self.network_predict_batch = self.lib.network_predict_batch
        self.network_predict_batch.argtypes = [
            ctypes.c_void_p, utils.IMAGE,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int
        ]
        self.network_predict_batch.restype = ctypes.POINTER(utils.DETNUMPAIR)

    def load_network(self, config, data, weights, batch_size=1, channels=3):
        self.network = self.load_net_custom(
            config.encode('ascii'),
            weights.encode('ascii'), 0,
            batch_size)
        metadata = self.load_meta(data.encode('ascii'))
        self.class_names = [metadata.names[i].decode('ascii') for i in range(metadata.classes)]

        self.resolution = (
            self.lib.network_width(self.network),
            self.lib.network_height(self.network)
        )

        self.image = self.make_image(
            self.lib.network_width(self.network),
            self.lib.network_height(self.network),
            channels
        )

    def detect(self, image, thresh=0.5, hier_thresh=0.5, nms=0.45):
        pnum = ctypes.pointer(ctypes.c_int(0))

        _image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _image = cv2.resize(_image, (self.image.w, self.image.h), interpolation=cv2.INTER_LINEAR)

        self.copy_image_from_bytes(self.image, _image.tobytes())
        self.predict_image(self.network, self.image)
        detections = self.get_network_boxes(
            self.network, self.image.w, self.image.h,
            thresh, hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        if nms:
            self.do_nms_sort(detections, num, len(self.class_names), nms)

        predictions = self.remove_negatives(detections, self.class_names, num)
        predictions = self.decode_detection(predictions)
        self.free_detections(detections, num)

        return sorted(predictions, key=lambda x: x[1])

    @staticmethod
    def remove_negatives(detections, class_names, num):
        predictions = []
        for i in range(num):
            for idx, name in enumerate(class_names):
                if detections[i].prob[idx] <= 0:
                    continue

                bbox = detections[i].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[i].prob[idx], bbox))

        return predictions

    @staticmethod
    def bbox2rect(bbox):
        x, y, w, h = bbox
        xmin = round(x - (w / 2))
        xmax = round(x + (w / 2))
        ymin = round(y - (h / 2))
        ymax = round(y + (h / 2))
        return xmin, ymin, xmax, ymax

    def class_colors(self):
        return {
            name: (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ) for name in self.class_names
        }

    @staticmethod
    def decode_detection(detections):
        decoded = []
        for label, confidence, bbox in detections:
            confidence = str(round(confidence * 100, 2))
            decoded.append((str(label), confidence, bbox))

        return decoded
