import numpy as np


def pooling(input, size=3, stride=1, padding=2, mode='max'):
    input = np.pad(array=input, pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    height = input.shape[2]
    width = input.shape[3]
    size_h = (height - size) // stride + 1
    size_w = (width - size) // stride + 1

    out = np.zeros((input.shape[0], input.shape[1], size_h, size_w))
    print(out.shape)

    for h in range(size_h):
        for w in range(size_w):
            if mode == 'average':
                out[:, :, h, w] = np.mean(input[:, :,
                                          h * stride: min(h * stride + size, height),
                                          w * stride: min(w * stride + size, width)], axis=(2, 3))
            if mode == 'max':
                out[:, :, h, w] = np.max(input[:, :,
                                         h * stride: min(h * stride + size, height),
                                         w * stride: min(w * stride + size, width)], axis=(2, 3))
    return out

if __name__ == '__main__':
    input = np.random.random((1, 1, 3, 3))
    out = pooling(input)
    print(input)
    print(out)
