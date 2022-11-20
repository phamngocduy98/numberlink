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

    def backtrack(self, i: int, N: int, selectedCount: int, maxSelectedCount: int, callback: Callable[[dict[int, bool]], None], selected: dict[int, bool] = {}):
        # print("selectedCount", selectedCount)

        if selectedCount == maxSelectedCount:
            # print("selectedCount == maxSelectedCount",
            #       selectedCount, maxSelectedCount)
            callback(selected)
            return
        if i == N:
            # print("i == N", i, N, selected)
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

        self.backtrack(0, len(var), 0, len(var) - k + 1, onBuildOK, {})
        return clauses

    def atMostK(self, k: int, var: list[int]) -> list[list[int]]:
        clauses = []

        def onBuildOK(selected: dict[int, bool]):
            vars = []
            for varIdx in selected:
                if selected[varIdx]:
                    vars.append(-var[varIdx])
            clauses.append(vars)

        self.backtrack(0, len(var), 0, k + 1, onBuildOK, {})
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


class Sudoku():
    def __init__(self, N: int, encoding: BinomialEncoding | None = None) -> None:
        self.N = N
        self.s = MinisatGH(use_timer=True)
        self.encoding = encoding or BinomialEncoding(N * N * N + 1)
        self.clausesCount = 0

    def index(self, i: int, j: int, k: int) -> int:
        return (i - 1) * (self.N**2) + (j - 1) * self.N + k

    def addClauses(self, clauses: list[list[int]]):
        for clause in clauses:
            # print("addClauses: add_clause", clause)
            self.s.add_clause(clause)
        self.clausesCount += len(clauses)

    # each box contains precisely one number
    def exactOneEachCell(self):
        # print("exactOneEachCell")
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                var = [self.index(i, j, k) for k in range(1, self.N + 1)]
                self.addClauses(self.encoding.exactOne(var))
                self.encoding.reset()

    # precisely once in each row
    def exactOneEachRow(self):
        # print("exactOneEachRow")
        for i in range(1, self.N + 1):
            for k in range(1, self.N + 1):
                var = [self.index(i, j, k) for j in range(1, self.N + 1)]
                self.addClauses(self.encoding.exactOne(var))
                self.encoding.reset()

    # precisely once in each row
    def exactOneEachCol(self):
        # print("exactOneEachCol")
        for j in range(1, self.N + 1):
            for k in range(1, self.N + 1):
                var = [self.index(i, j, k) for i in range(1, self.N + 1)]
                self.addClauses(self.encoding.exactOne(var))
                self.encoding.reset()

    # precisely once in each sub-sqrt(N)xsqrt(N) matrix
    def exactOneEachSubMatrix(self):
        # print("exactOneEachSubMatrix")
        sqrtN = int(math.sqrt(self.N))

        #  find subsize
        subSizeI = 0
        subSizeJ = 0
        for i in range(2, sqrtN + 1):
            j = sqrtN + 2 - i
            if self.N % j == 0:
                subSizeI = j
                subSizeJ = self.N // j
                break
        if subSizeI == 0 or subSizeJ == 0:
            print("WARN: No sub-matrix rule. SAT solver may take longer time to solve")
            return

        # print(f"subSize {subSizeI}x{subSizeJ}")
        for k in range(1, self.N + 1):
            for subi in range(0, subSizeJ):
                for subj in range(0, subSizeI):
                    var = [self.index(
                        subi * subSizeI + i, subj * subSizeJ + j, k) for i in range(1, subSizeI + 1) for j in range(1, subSizeJ + 1)]
                    self.addClauses(self.encoding.exactOne(var))
                    self.encoding.reset()

    def initialValue(self, data: list[list[int | None]]):
        for (i, row) in enumerate(data):
            for (j, cell) in enumerate(row):
                if cell:
                    # print("initialValue, add_clause", i+1, j+1, cell, [
                    #       self.index(i+1, j+1, cell)])
                    self.s.add_clause([self.index(i+1, j+1, cell)])

    def solve(self):
        startTime = time()
        self.exactOneEachCell()
        self.exactOneEachRow()
        self.exactOneEachCol()
        self.exactOneEachSubMatrix()
        buildTime = time()

        self.s.solve()
        solveTime = time()
        self.model = self.s.get_model()
        # print("[Solve] Result = ", self.model)
        newVarCount = self.encoding.varStartIdx - self.N * self.N * self.N - 1
        satSolveTime = self.s.time()
        # print('[Solve] Build {0} clauses, {1} new variables in {2:.2f}s'.format(self.clausesCount, newVarCount,
        #                                                                         buildTime - startTime))
        # print('[Solve] Solved in {0:.2f}s'.format(satSolveTime))
        # print('[Solve] Total solved in {0:.2f}s'.format(solveTime - startTime))
        self.s.delete()
        return self.clausesCount, newVarCount, buildTime - startTime, satSolveTime, solveTime - startTime

    def printResult(self):
        if not self.model:
            print("No Solution")
            return
        for i in range(1, self.N + 1):
            line = ""
            for j in range(1, self.N + 1):
                val = None
                for k in range(1, self.N + 1):
                    if self.model[self.index(i, j, k) - 1] > 0:
                        if val:
                            line += ("+%s" % k)
                        else:
                            line += ("%s" % k)
                        val = k
                if not val:
                    line += "x"
                line += ","
            print(f"[{line}],")

    def drawResult(self):
        if not self.model:
            print("No Solution")
            return

        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                       for i in range(self.N + 1)]
        figSizeW = 0.7*self.N
        figSizeH = 0.7*self.N
        cellSize = 0.7/figSizeW

        fig, ax1 = plt.subplots(figsize=(figSizeW, figSizeH))
        ax1.axis('off')
        table = ax1.table(
            cellColours=[["0"], ["0"]], loc='center', cellLoc='center')
        table.set_fontsize(12)

        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                val = None
                cellVal = ""
                for k in range(1, self.N + 1):
                    if self.model[self.index(i, j, k) - 1] > 0:
                        if val:
                            cellVal += ("+%s" % k)
                        else:
                            cellVal += ("%s" % k)
                        val = k
                if not val:
                    cellVal += "x"
                table.add_cell(row=i-1, col=j-1, width=cellSize, height=cellSize,
                               text=cellVal, loc="center", facecolor=self.colors[val] or "w")
        plt.savefig("sodoku.png", dpi=200, bbox_inches='tight')


class NumberLink():
    def __init__(self, N: int, encoding: BinomialEncoding | None = None) -> None:
        self.N = N
        self.s = MinisatGH(use_timer=True)
        self.encoding = encoding or BinomialEncoding(N * N * N + 1)
        self.clausesCount = 0

        self.numberCount = 0
        self.initTable = [[]]  # type: list[list[int]]

    def index(self, i: int, j: int, k: int) -> int:
        return (i - 1) * (self.N**2) + (j - 1) * self.N + k

    def fourNeighbor(self, i: int, j: int, k: int):
        di = [1, 0, -1, 0]
        dj = [0, 1, 0, -1]
        vars = []
        for ii in di:
            for jj in dj:
                _i = i+ii
                _j = j+jj
                if _i > 0 and _i <= self.N and _j > 0 and _j <= self.N:
                    vars.append(self.index(_i, _j, k))
                    print(f"fourNeighbor{i}{j}{k} = {_i}{_j}")
        print("fourNeighbor", vars)
        return vars

    def addClauses(self, clauses: list[list[int]]):
        for clause in clauses:
            # print("addClauses: add_clause", clause)
            self.s.add_clause(clause)
        self.clausesCount += len(clauses)

    def exactOneEachCell(self):
        print("exactOneEachCell")
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                var = [self.index(i, j, k) for k in range(1, self.N + 1)]
                self.addClauses(self.encoding.exactOne(var))
                self.encoding.reset()

    def exactOneDirection(self):
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                if self.initTable[i-1][j-1] is not None:
                    # isEdgeNode
                    k = self.initTable[i-1][j-1]
                    cl = self.encoding.atMostOne(self.fourNeighbor(i, j, k))
                    self.addClauses(cl)
                else:
                    pass
                    # is<Unknown>Vertex
                    # for k in range(1, self.numberCount + 1):
                    #     cl = self.encoding.exactK(
                    #         2, self.fourNeighbor(i, j, k))

    def printResult(self):
        if not self.model:
            print("No Solution")
            return
        for i in range(1, self.N + 1):
            line = ""
            for j in range(1, self.N + 1):
                val = None
                for k in range(1, self.N + 1):
                    if self.model[self.index(i, j, k) - 1] > 0:
                        if val:
                            line += ("+%s" % k)
                        else:
                            line += ("%s" % k)
                        val = k
                if not val:
                    line += "x"
                line += ","
            print(f"[{line}],")

    def solve(self, initTable: list[list[int]]):
        self.initTable = initTable
        for i, row in enumerate(initTable):
            for j, k in enumerate(row):
                if k is not None:
                    self.numberCount += 1
                    self.s.add_clause([self.index(i+1, j+1, k+1)])

        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                       for i in range(self.numberCount)]

        startTime = time()
        # self.exactOneEachCell()
        # self.exactOneDirection()

        buildTime = time()

        self.s.solve()
        solveTime = time()
        self.model = self.s.get_model()
        # print("[Solve] Result = ", self.model)
        newVarCount = self.encoding.varStartIdx - self.N * self.N * self.N - 1
        print('[Solve] Build {0} clauses, {1} new variables in {2:.2f}s'.format(self.clausesCount, newVarCount,
                                                                                buildTime - startTime))
        print('[Solve] Solved in {0:.2f}s'.format(self.s.time()))
        print('[Solve] Total solved in {0:.2f}s'.format(solveTime - startTime))
        self.s.delete()

    def drawResult(self):
        if not self.model:
            print("No Solution")
            return

        figSizeW = 0.7*self.N
        figSizeH = 0.7*self.N
        cellSize = 0.7/figSizeW

        fig, ax1 = plt.subplots(figsize=(figSizeW, figSizeH))
        ax1.axis('off')
        table = ax1.table(
            cellColours=[["0"], ["0"]], loc='center', cellLoc='center')
        table.set_fontsize(12)

        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                val = None
                cellVal = ""
                for k in range(1, self.N + 1):
                    if self.model[self.index(i, j, k) - 1] > 0:
                        if val:
                            cellVal += ("+%s" % k)
                        else:
                            cellVal += ("%s" % k)
                        val = k
                if not val:
                    cellVal += "x"
                table.add_cell(row=i-1, col=j-1, width=cellSize, height=cellSize,
                               text=cellVal, loc="center", facecolor=self.colors[val] or "w")
        plt.savefig("numberlink.png", dpi=200, bbox_inches='tight')


# if __name__ == "__main__":
#     binomial = BinomialEncoding(100)
#     print(binomial.atMostK(2, [1, 2, 3, 4, 5]))
#     print(binomial.atLeastK(2, [1, 2, 3, 4, 5]))
