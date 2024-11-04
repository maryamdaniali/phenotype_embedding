import numpy as np

class Sparse2D():
    
    def __init__(self, _n):
        if isinstance(_n, int):
            self._vector = np.zeros(_n)
            self._matrix = np.zeros(self.__lenArray(_n))

    def __len__(self): 
        return len(self._matrix)

    def __lenArray(self, _n):
        _len = _n 
        for _j in range(1, _n):
            self._vector[_j] = _len 
            for _i in range(_j, _n):
                _len += 1
        return _len
    
    def __pos(self, i, j):
        if i > j:
            i,j = j,i
        p = int(self._vector[i]) + int(j-i) 

        return p

    def set(self, i, j, n):
        p = self.__pos(i, j) 
        #print("position i {}, j {} is at post {}". format(i,j,p))
        self._matrix[p] = n

    def get(self, i, j):
        p = self.__pos(i, j) 
        return self._matrix[p]