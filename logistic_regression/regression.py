# %load regression.py
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

    alpha, epochs = 0.001, 500
    data = np.column_stack((np.ones((batch_size, 1)), data))    # 为Bias最后一列添加1,X = [X|1]
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

def show(weights):
    import matplotlib.pyplot as plt
    data, labels = get_data()
    
    Xcord_P, Ycord_P = [], []
    Xcord_M, Ycord_M = [], []
    for i in range(len(data)):
        if labels[i][0] == 1:
            Xcord_P.append(data[i][0])
            Ycord_P.append(data[i][1])
        else:
            Xcord_M.append(data[i][0])
            Ycord_M.append(data[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Xcord_P, Ycord_P, s=30, c="red", marker="s")
    ax.scatter(Xcord_M, Ycord_M, s=30, c="green")
    x = np.arange(-3.0, 3.0, 0.1)
    # ln(p/(1-p)) = W0 + W1*x + W2*y
    # 当p=0.5时，ln(p/(1-p))=0
    # 所以y = (-W0-W1*x) / W2
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    # getA() matrix => array
    data, labels = get_data()
    weights = gradient_ascent(data, labels)
    print(gradient_ascent(data, labels))
    show(weights.getA())