# BN 前向计算主要把握归一化的维度，以及归一化公式xi'=(xi-means)/sqrt(var+eps), xi''=gamma*xi'+beta

# 不调用库函数
def bn_forward(input, gamma=1.0, beta=0.0, eps=1e-4):
    N, C, W, H = len(input), len(input[0]), len(input[0][0]), len(input[0][0][0])
    output = input.copy()

    for c in range(C):
        for n in range(N):
            sums = 0.0
            for w in range(W):
                for h in range(H):
                    sums += input[n][c][w][h]
        means = sums / (N * W * H)

        for n in range(N):
            sums = 0.0
            for w in range(W):
                for h in range(H):
                    var = (input[n][c][h][w] - means) ** 2
                    import math # 使用开方函数，面试应该允许使用
                    output[n][c][h][w] = gamma + ((input[n][c][h][w] - means) / math.sqrt(var + eps)) + beta

    return output