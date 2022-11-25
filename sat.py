from time import time
from pysat.solvers import MinisatGH
from typing import Callable
import math
import matplotlib.pyplot as plt
import random


class Encoding():
    def __init__(self, varStartIdx: int) -> None:
        self.varStartIdx = varStartIdx
        self.newVarCount = 0

    def reset(self):
        self.varStartIdx += self.newVarCount

    def atLeastOne(self, var: list[int]) -> list[list[int]]:
        return []

    def atMostOne(self, var: list[int]) -> list[list[int]]:
        return []

    def atLeastK(self, k: int, var: list[int]) -> list[list[int]]:
        return []

    def atMostK(self, k: int, var: list[int]) -> list[list[int]]:
        return []

    def exactOne(self, var: list[int]) -> list[list[int]]:
        # print("exactOne", var)
        clauses = self.atMostOne(var)
        clauses.extend(self.atLeastOne(var))
        return clauses

    def exactK(self, k: int, var: list[int]) -> list[list[int]]:
        # print("exactOne", var)
        clauses = self.atMostK(k, var)
        clauses.extend(self.atLeastK(k, var))
        return clauses


class BinomialEncoding(Encoding):
    backtrackCount = 0
    def atLeastOne(self, var: list[int]) -> list[list[int]]:
        # print("atLeastOne", var)
        return [[var[k] for k in range(0, len(var))]]

    def atMostOne(self, var: list[int]) -> list[list[int]]:
        # print("atMostOne", var)
        clauses = []
        for i in range(0, len(var)):
            for j in range(0, len(var)):
                if i != j:
                    clauses.append([-var[i], -var[j]])
        return clauses

    def startBacktrack(self,N:int, maxSelectedCount:int, callback:Callable[[dict[int, bool]], None]):
        self.backtrackCount = 0
        self.backtrack(0,N,0, maxSelectedCount, callback,{})

    def backtrack(self, i: int, N: int, selectedCount: int, maxSelectedCount: int, callback: Callable[[dict[int, bool]], None], selected: dict[int, bool] = {}):
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

    def atLeastK(self, k: int, var: list[int]) -> list[list[int]]:
        clauses = []

        def onBuildOK(selected: dict[int, bool]):
            vars = []
            for varIdx in selected:
                if selected[varIdx]:
                    vars.append(var[varIdx])
            clauses.append(vars)

        self.startBacktrack( len(var),  len(var) - k + 1, onBuildOK)
        return clauses

    def atMostK(self, k: int, var: list[int]) -> list[list[int]]:
        clauses = []

        def onBuildOK(selected: dict[int, bool]):
            vars = []
            for varIdx in selected:
                if selected[varIdx]:
                    vars.append(-var[varIdx])
            clauses.append(vars)

        self.backtrack(len(var), k + 1, onBuildOK)
        return clauses


class BinaryEncoding(Encoding):
    def __init__(self, varStartIdx: int) -> None:
        Encoding.__init__(self, varStartIdx)

    def Yj(self, i: int):
        return self.varStartIdx + i

    def atLeastK(self, k: int, var: list[int]) -> list[list[int]]:
        return []

    def atMostK(self, k: int, var: list[int]) -> list[list[int]]:
        return []

    def atLeastOne(self, var: list[int]) -> list[list[int]]:
        # print("atLeastOne", var)
        return [[var[k] for k in range(0, len(var))]]

    def atMostOne(self, var: list[int]) -> list[list[int]]:
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
