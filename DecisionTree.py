import numpy as np
import math

def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)


class DecisionTree():
    def __init__(self, x, y, idxs, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y.values[idxs])
        self.score = float('inf')
        self.find_varsplit()

    def find_varsplit(self):
        for i in range(self.c): self.find_better_split(i)
        if self.score == float('inf'): return
        x = self.split_col
        leftIndex = np.nonzero(x <= self.splitValue)[0]
        rightIndex = np.nonzero(x > self.splitValue)[0]
        self.left = DecisionTree(self.x, self.y, self.idxs[leftIndex])
        self.right = DecisionTree(self.x, self.y, self.idxs[rightIndex])

    def find_better_split(self, var_idx):
        x, y = self.x.values[self.idxs, var_idx], self.y.values[self.idxs]
        sortedIndices = np.argsort(x)

        sort_y, sort_x = y[sortedIndices], x[sortedIndices]

        rightCount, rightSum, rightSumSquared = self.n, sort_y.sum(), (sort_y ** 2).sum()
        leftCount, leftSum, leftSumSquared = 0, 0., 0.

        for i in range(0, self.n - self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]

            leftCount += 1;
            rightCount -= 1

            leftSum += yi;
            rightSum -= yi

            leftSumSquared += yi ** 2;
            rightSumSquared -= yi ** 2

            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            leftSTD = std_agg(leftCount, leftSum, leftSumSquared)
            rightSTD = std_agg(rightCount, rightSum, rightSumSquared)
            curr_score = leftSTD * leftCount + rightSTD * rightCount
            
            if curr_score < self.score:
                self.column, self.score, self.splitValue = var_idx, curr_score, xi

    @property
    def split_name(self):
        return self.x.columns[self.column]

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.column]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.splitValue}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.left if xi[self.column] <= self.splitValue else self.right
        return t.predict_row(xi)

