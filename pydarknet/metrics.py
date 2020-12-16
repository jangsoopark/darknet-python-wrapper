import pydarknet.darknet


def iou(a, b):
    # print(b)
    def intersect():
        xmin = max(a[0], b[0])
        ymin = max(a[1], b[1])
        xmax = min(a[2], b[2])
        ymax = min(a[3], b[3])
        return xmin, ymin, xmax, ymax

    def area(box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        return w * h

    ibox = intersect()
    if ibox[2] - ibox[0] <= 0 or ibox[3] - ibox[1] <= 0:
        return 0

    intersection = area(ibox)
    union = area(a) + area(b) - intersection
    return intersection / union


def confusion_metrics(gt, pred, class_names, resolution, iou_thresh=.5):
    w, h = resolution

    confusions = {
        name: {'tp': 0, 'fp': 0, 'fn': 0}
        for name in class_names
    }

    pred = sorted(pred, key=lambda x: x[1])
    pred = [(p[0], p[1], pydarknet.darknet.Darknet.bbox2rect(p[2])) for p in pred]

    gt[:, 1::2] = gt[:, 1::2] * w
    gt[:, 2::2] = gt[:, 2::2] * h
    gt = [(class_names[int(e[0])], pydarknet.darknet.Darknet.bbox2rect(e[1:])) for e in gt.tolist()]

    for g in gt:
        class_name = g[0]
        gbox = g[1]

        if not len(pred):  # false negative
            confusions[class_name]['fn'] += 1
            continue

        match = max(pred, key=lambda x: iou(gbox, x[2]) * (class_name == x[0]))
        if not match[0] == class_name:
            continue

        if iou(gbox, match[2]) < iou_thresh:
            continue

        del pred[pred.index(match)]

        confusions[class_name]['tp'] += 1

    for p in pred:
        class_name = p[0]
        confusions[class_name]['fp'] += 1

    return confusions


def accuracy(confusion):
    try:
        return confusion['tp'] / (confusion['tp'] + confusion['fp'] + confusion['fn'])
    except ZeroDivisionError:
        return 1


def precision(confusion):
    try:
        return confusion['tp'] / (confusion['tp'] + confusion['fp'])
    except ZeroDivisionError:
        return 1


def recall(confusion):
    try:
        return confusion['tp'] / (confusion['tp'] + confusion['fn'])
    except ZeroDivisionError:
        return 1


def f1_score(confusion):
    p = precision(confusion)
    r = recall(confusion)
    try:
        return 2 * p * r / (p + r)
    except ZeroDivisionError:
        return 0


def show(gt, pred, class_names, resolution, iou_thresh=.5):
    confusions = confusion_metrics(gt, pred, class_names, resolution, iou_thresh)

    for name, c in confusions.items():
        p = precision(c)
        r = recall(c)
        f1 = f1_score(c)
        print('{:22s} precision: {:.4f} recall: {:.4f} F1-score: {:.4f}'.format(name, p, r, f1))
        print('{:22s} TP: {:3d} FP: {:3d} FN: {:3d}'.format('', c['tp'], c['fp'], c['fn']))
    print()
