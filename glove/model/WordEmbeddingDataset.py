import torch.utils.data as tud
import torch

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, co_matrix, weight_matrix):
        self.co_matrix = co_matrix
        self.weight_matrix = weight_matrix
        self.train_set = []

        for i in range(self.weight_matrix.shape[0]):
            for j in range(self.weight_matrix.shape[1]):
                if weight_matrix[i][j] != 0:
                    # 这里对权重进行了筛选，去掉权重为0的项
                    # 因为共现次数为0会导致log(X)变成nan
                    self.train_set.append((i, j))

    def __len__(self):
        '''
        必须重写的方法
        :return: 返回训练集的大小
        '''
        return len(self.train_set)

    def __getitem__(self, index):
        '''
        必须重写的方法
        :param index:样本索引
        :return: 返回一个样本
        '''
        (i, j) = self.train_set[index]
        return i, j, torch.tensor(self.co_matrix[i][j], dtype=torch.float), self.weight_matrix[i][j]
