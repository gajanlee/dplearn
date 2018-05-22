import numpy as np

def convert_matrix(func):
    def convert(data, labels):
        return func(np.mat(data), np.mat(labels))
    return convert

def sigmoid(output):
    return 1 / (1 + np.exp(-output))

@convert_matrix
def gradient_ascent(data, labels):
    batch_size, feature_size = data.shape
    print(data.shape, labels.shape)
    assert labels.shape == (batch_size, 1)

    alpha, epochs = 0.001, 80000
    data = np.column_stack((data, np.ones((batch_size, 1))))    # 为Bias最后一列添加1,X = [X|1]
    weights = np.ones((feature_size + 1, 1))    # 可以尝试随机初始化，添加一列作为bias
    for _ in range(epochs):
        output = sigmoid(data * weights)
        error = (labels - output)
        weights = weights + alpha * data.transpose() * error
    return weights

def get_data():
    data, labels = [], []
    with open("data.txt") as fp:
        for line in fp:
            d = line.strip().split()
            data.append(list(map(float, d[:-1])))
            labels.append([float(d[-1])])
    return data, labels

if __name__ == "__main__":
    # getA() matrix => array
    data, labels = get_data()
    print(data[0], type(labels[0]))
    print(gradient_ascent(data, labels))