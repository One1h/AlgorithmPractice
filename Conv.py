from PIL import Image
import numpy as np


def Conv(input, k_size=3, padding=0, stride=1):
    width, height, channel = input.shape
    output = np.zeros((Output_Size(width, k_size, padding, stride), Output_Size(width, k_size, padding, stride), channel))

    for c in range(channel):
        # 随机卷积核
        # kernel = np.random.randint(low=0, high=3, size=(k_size, k_size))
        #固定卷积核
        kernel = np.array([[0,1,2],[2,2,0],[0,1,2]])

        input_c = input[:, :, c]
        # padding操作
        inp_pad = np.pad(array=input_c, pad_width=((padding, padding), (padding, padding)), mode='constant', constant_values=0)

        output_c = np.zeros(inp_pad.shape)
        half_size = int(k_size / 2)
        # 遍历计算卷积
        for x in range(half_size, width + 2 * padding - half_size):
            for y in range(half_size, width + 2 * padding - half_size):
                output_c[x][y] = np.sum(np.multiply(kernel, inp_pad[x-half_size: x+half_size+1, y-half_size:y+half_size+1]))

        output[:, :, c] = output_c[half_size:output_c.shape[0] - half_size, half_size:output_c.shape[1] - half_size]
    return output


# 计算输出尺寸
def Output_Size(W, K, P, S):
    return int((W + 2 * P - K)/S + 1)


if __name__ =='__main__':
    # img = Image.open('1.jpg')
    # input = np.array(img)

    input = np.array([[[3], [3], [2], [1], [0]],
                      [[0], [0], [1], [3], [1]],
                      [[3], [1], [2], [2], [3]],
                      [[2], [0], [0], [2], [2]],
                      [[2], [0], [0], [0], [1]]])
    output = Conv(input)
    print(output)
