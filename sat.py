import math
import random
from time import time
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
from pysat.solvers import MinisatGH


class Encoding():
    def __init__(self, varStartIdx):
        self.varStartIdx = varStartIdx
        self.newVarCount = 0

    def reset(self):
        self.varStartIdx += self.newVarCount

    def atLeastOne(self, var) :
        return []

    def atMostOne(self, var) :
        return []

    def atsLeatK(self, k, var) :
        return []

    def atMostK(self, k, var) :
        return []

    def exactOne(self, var) :
        # print("exactOne", var)
        clauses = self.atMostOne(var)
        clauses.extend(self.atLeastOne(var))
        return clauses

    def exactK(self, k, var) :
        # print("exactOne", var)
        clauses = self.atMostK(k, var)
        clauses.extend(self.atLeastK(k, var))
        return clauses


class BinomialEncoding(Encoding):
    backtrackCount = 0
    def atLeastOne(self, var) :
        # print("atLeastOne", var)
        return [[var[k] for k in range(0, len(var))]]

    def atMostOne(self, var) :
        # print("atMostOne", var)
        clauses = []
        for i in range(0, len(var)):
            for j in range(0, len(var)):
                if i != j:
                    clauses.append([-var[i], -var[j]])
        return clauses

    def startBacktrack(self,N, maxSelectedCount, callback):
        self.backtrackCount = 0
        self.backtrack(0,N,0, maxSelectedCount, callback,{})

    def backtrack(self, i, N, selectedCount, maxSelectedCount, callback, selected = {}):
        # print("selectedCount", selectedCount)
        if self.backtrackCount > 10000:
            raise Exception("Backtrack call limit exceeded")
        self.backtrackCount +=1

        if selectedCount == maxSelectedCount:
            # print("selectedCount == maxSelectedCount",
            #       selectedCount, maxSelectedCount)
            callback(selected)
            return
        if i == N:
            # print("i == N", i, N, selected)
            return
        if N - i < maxSelectedCount - selectedCount:
            return
        selected[i] = True
        self.backtrack(i+1, N, selectedCount+1,
                       maxSelectedCount, callback, selected)
        selected[i] = False
        self.backtrack(i+1, N, selectedCount,
                       maxSelectedCount, callback, selected)

    def atLeastK(self, k, var) :
        clauses = []

        def onBuildOK(selected):
            vars = []
            for varIdx in selected:
                if selected[varIdx]:
                    vars.append(var[varIdx])
            clauses.append(vars)

        self.startBacktrack( len(var),  len(var) - k + 1, onBuildOK)
        return clauses

    def atMostK(self, k, var) :
        clauses = []

        def onBuildOK(selected):
            vars = []
            for varIdx in selected:
                if selected[varIdx]:
                    vars.append(-var[varIdx])
            clauses.append(vars)

        self.startBacktrack(len(var), k + 1, onBuildOK)
        return clauses


class BinaryEncoding(Encoding):
    def __init__(self, varStartIdx) -> None:
        Encoding.__init__(self, varStartIdx)

    def Yj(self, i):
        return self.varStartIdx + i

    def atLeastK(self, k, var) :
        return []

    def atMostK(self, k, var) :
        return []

    def atLeastOne(self, var) :
        # print("atLeastOne", var)
        return [[var[k] for k in range(0, len(var))]]

    def atMostOne(self, var) :
        # print("atMostOne", var)
        clauses = []
        N = len(var)
        self.newVarCount = math.ceil(math.log2(N))

        for i in range(0, N):
            j = self.newVarCount - 1  # Y[j]
            xi = var[i]
            while i > 0 or j >= 0:
                jbit = i % 2  # j-th bit of i (right to left)
                clauses.append([-xi, self.Yj(j) *
                               (1 if jbit == 1 else -1)])
                j -= 1
                i //= 2
        return clauses

# if __name__ == "__main__":
#     binomial = BinomialEncoding(100)
#     print(binomial.atMostK(2, [1, 2, 3, 4, 5]))
#     print(binomial.atLeastK(2, [1, 2, 3, 4, 5]))
