import numpy as np


def pooling(input, size=2, mode='max'):
    height = input.shape[2]
    width = input.shape[3]
    size_h = int(np.ceil(height / size))
    size_w = int(np.ceil(width / size))

    out = np.zeros((input.shape[0], input.shape[1], size_h, size_w))

    for h in range(size_h):
        for w in range(size_w):
            if mode == 'average':
                out[:, :, h, w] = np.mean(input[:, :,
                                          h * size : min((h + 1) * size, height),
                                          w * size : min((w + 1) * size, width)])
            if mode == 'max':
                out[:, :, h, w] = np.max(input[:, :,
                                          h * size : min((h + 1) * size, height),
                                          w * size : min((w + 1) * size, width)])
    return out


if __name__ == '__main__':
    input = np.random.random((1, 1, 5, 5))
    out = pooling(input)
    print(input)
    print(out)
