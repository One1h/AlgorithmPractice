import numpy as np

def NMS(dets, thresh=0.5):
    # 截取五个参数
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:,4]

    # 计算每个Bbox的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按scores从高到低排序
    order = scores.argsort()[::-1]

    # 保留的bbox
    candidate = []
    while order.size > 0:
        index = order[0]
        candidate.append(index)

        # 计算交集框的重叠部分的左上和右下坐标
        x_min = np.maximum(x1[index], x1[order[1:]])
        y_min = np.maximum(y1[index], y1[order[1:]])
        x_max = np.minimum(x2[index], x2[order[1:]])
        y_max = np.minimum(y2[index], y2[order[1:]])

        # 获取交集部分宽高，若无交集则为0
        w = np.maximum(0.0, x_max - x_min + 1)
        h = np.maximum(0.0, y_max - y_min + 1)

        # 计算IoU
        iou = (w * h) / (areas[index] + areas[order[1:]] - (w * h))
        inds = np.where(iou <= thresh)[0]

        # 对IoU小于阈值的进行下一步NMS
        order = order[inds + 1]

    return candidate


if __name__ == '__main__':
    dets = [[50, 50, 100, 100, 0.6],
             [40, 40, 90, 90, 0.7],
             [45, 45, 80, 80, 0.2],
             [55, 55, 110, 110, 0.3]]
    dets = np.array(dets)
    index = NMS(dets, 0.1)
    print(dets[index])