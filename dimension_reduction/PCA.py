import numpy as np
from dplearn.datasets.dimension_reduction import _3dim_polar


def batch_PCA(data_matrix, target_dim):
    pass


def central(data_matrix):
    meanVal = np.mean(data_matrix, axis=0)
    return data_matrix - meanVal, meanVal

def PCA(data_matrix, target_dim=2):
    # batch_size, dims = data_matrix
    data_matrix, mean_matrix = central(data_matrix)
    covMat = np.cov(data_matrix, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    #k = np.eigValPct(eigVals, 0.9)
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(target_dim+1):-1]
    redEigVects = eigVects[:,eigValInd]
    
    lowDataMat = data_matrix * redEigVects
    reconMat = (lowDataMat*redEigVects.T) + mean_matrix
    return lowDataMat, reconMat



def display(data_matrix):
    import matplotlib.pyplot as plt
    Xcords, Ycords = [], []
    for data in data_matrix:
        data = data.getA()[0]
        Xcords.append(data[0])
        Ycords.append(data[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Xcords, Ycords, s=30, c="red", marker="s")
    plt.show()

if __name__ == "__main__":
    _d, data = PCA(_3dim_polar())
    display(_d)
    pass