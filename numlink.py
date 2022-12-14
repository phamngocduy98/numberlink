import csv
import math
import time
import traceback
import matplotlib.pyplot as plt
import random
from pysat.solvers import Solver
from tabulate import tabulate
from sat import BinaryEncoding, BinomialEncoding

class NumberLink:
    def __init__(self, input):
        self.input = input
        self.ROW = len(input)
        self.COL = len(input[0])
        self.Label_count = 0
        self.Labels = []
        self.LabelStartPos = {}
        self.LabelEndPos={}
        for row in range(self.ROW):
            for col in range(self.COL):
                if (input[row][col] >= 0 ):
                    label = input[row][col]
                    if input[row][col] not in self.Labels:
                        self.Labels.append(label)
                        self.LabelStartPos[label] = row,col
                    else:
                        self.LabelEndPos[label] = row,col
        self.Label_count = len(self.Labels)
        # print("Label", self.Labels)
        self.MAX_VALUE = self.VAR_COUNT = self.ROW * self.COL * self.Label_count
        self.colors = ["#"+''.join([random.choice('23456789ABCDEF') for j in range(6)])
                       for i in range(self.Label_count)]

    def _draw(self, matrix, fileName=None):
        if (len(matrix)):
            dpi = 100
            figSizeW = max(self.COL*0.15,2)
            figSizeH = max(self.ROW*0.15,2)
            cellSizeW = figSizeW/self.COL
            cellSizeH = figSizeH/self.ROW
            fig, ax1 = plt.subplots(figsize=(figSizeW, figSizeH))
            ax1.axis('off')

            table = ax1.table(
                cellColours=[["0"], ["0"]], loc='center', cellLoc='center')
            for row in range(self.ROW):
                for col in range(self.COL):
                    num = matrix[row][col]
                    idx = self.Labels.index(num) if num in self.Labels else -1
                    table.add_cell(row=row, col=col, width=cellSizeW, height=cellSizeH,
                                   text=f"{idx}" if idx >-1 else '', loc="center", facecolor=self.colors[idx] if idx >= 0 else "w")
            table.set_fontsize(min(cellSizeW*dpi*0.8,10))
            plt.savefig("outputs/"+(fileName or "numberlink.png"),
                        dpi=dpi, bbox_inches='tight')
            plt.close()

    def draw(self, fileName=None):
        self._draw(self.input, f"{fileName}_input.png")
        self._draw(self.result, f"{fileName}_output.png")


    def get_index(self, i, j, k):
        return self.Label_count*(self.COL*i + j) + k + 1

    def get_loc(self, idx):
        idx = idx - 1
        k = idx % self.Label_count
        idx = idx // self.Label_count
        j = idx % self.COL
        i = idx // self.COL
        return (i, j, self.Labels[k])

    def get_number_link_matrix(self, sat_solve_result):
        number_link_matrix = [[-1]*self.COL for i in range(self.ROW)]
        for var in sat_solve_result:
            if (self.idx_in_matrix(var)):
                i, j, k = self.get_loc(var)
                number_link_matrix[i][j] = k
        return number_link_matrix

    def removeCircle(self, matrix):
        dx=[1,-1,0,0]
        dy=[0,0,1,-1]
        resultMatrix = [[-1 for i in range(len(matrix[0]))] for j in range(len(matrix))]
        def visit(row,col,label,endR,endC):
            if not visited[row][col] and matrix[row][col]==label:
                visited[row][col] = True
                resultMatrix[row][col]= matrix[row][col]
                if row == endR and col == endC :
                    return
                else:
                    for i in range(4):
                        r = row+dy[i]
                        c = col+dx[i]
                        if self.in_matrix(r,c):
                            visit(r, c,label, endR, endC)

        for label in self.LabelStartPos:
            row,col = self.LabelStartPos[label]
            endR, endC = self.LabelEndPos[label]
            visited =[[False for i in range(len(matrix[0]))] for j in range(len(matrix))]
            visit(row,col,label,endR, endC)
        return resultMatrix

        


    def idx_in_matrix(self, idx):
        return idx > 0 and idx <= self.MAX_VALUE

    def in_matrix(self, row, col):
        return row >= 0 and row < self.ROW and col >= 0 and col < self.COL

    def list_neighbor_cell_index(self, row, col, num, without_fixed_cell=False):
        dr = [1, -1, 0, 0]
        dc = [0, 0, 1, -1]
        nbr_cells = []
        for i in range(4):
            ri = row + dr[i]
            ci = col + dc[i]
            if (self.in_matrix(ri, ci)):
                if not without_fixed_cell or (without_fixed_cell and self.input[ri][ci] == -1):
                    idx = self.get_index(ri, ci, num)
                    nbr_cells.append(idx)
        return nbr_cells

    def solve(self):
        solver = Solver(name="minisat22", use_timer=True, bootstrap_with=[])
        binomial = BinomialEncoding(1)
        self.startTime = time.time()
        for row in range(self.ROW):
            for col in range(self.COL):
                labels = [self.get_index(row, col, num)
                          for num in range(self.Label_count)]
                rules = binomial.exactOne(labels)  # Ch??nh x??c 1 nh??n ???????c ch???n cho 1 ??
                solver.append_formula(rules)

                if (self.input[row][col] >= 0):  # Nh??n ???????c ??i???n s???n
                    num = self.Labels.index(self.input[row][col])
                    label = self.get_index(row, col, num)
                    solver.add_clause([label])

                    # Ch??nh x??c 1 ?? trong 4 ?? xung quanh ???????c ??i???n nh??n gi???ng n??
                    vars = self.list_neighbor_cell_index(
                        row, col, num)
                    rules = binomial.exactOne(vars)
                    solver.append_formula(rules)
                else:
                    for num in range(self.Label_count):
                        label = self.get_index(
                            row, col, num)  # Nh??n gi??? ?????nh
                        # L???y danh s??ch bi???n c??c ?? xung quanh v???i nh??n gi???ng nh??n gi??? ?????nh
                        nbr_label_vars = self.list_neighbor_cell_index(
                            row, col, num)

                        # N???u nh??n gi??? ?????nh ???????c ch???n -> ch??nh x??c 2 ?? xung quanh ???????c g??n nh??n gi???ng n??
                        # Ch??nh x??c 2 ?? xung quanh c?? nh??n gi???ng nh??n gi??? ?????nh // Lu???t t???m theo nh??n gi??? ?????nh
                        rules_cnf = binomial.exactK(2, nbr_label_vars)
                        # label -> rules_cnf  <=> -label V rules <=> -label V (rule1 ^ rule2 ^ ....) <=> (-label V rule1) ^ (-label V rule2) .....
                        for rule in rules_cnf:
                            solver.append_formula([[-label] + rule])
        self.VAR_COUNT = solver.nof_vars()
        self.CLAUSES = solver.nof_clauses()
        self.buildTime = time.time()
        solver.solve()
        self.accTime = solver.time_accum()
        result = solver.get_model()
        self.satTime = time.time()
        self.result =self.removeCircle(self.get_number_link_matrix(result) if result else [])
        return self.result

    def getBuildTime(self):
        return self.buildTime - self.startTime

    def getSATTime(self):
        return self.accTime

    def getTotalTime(self):
        return self.satTime - self.startTime


def toNumber(c):
    ordc = ord(c)
    if 48 <= ordc and ordc <= 57:  # isNumber [0-9]
        return ordc - 48
    if 65 <= ordc and ordc <= 90:  # isUpperChar [A-Z] [10-35]
        return 10 + ordc - 65
    if 97 <= ordc and ordc <= 122:  # isLowerChar [a-z] [36-61]
        return 36 + ordc - 97
    return -1


def readFile(fileName):
    questions = []

    file = open(fileName)
    line = " "
    while True:
        line = file.readline()
        if len(line) == 0:
            break

        if line.startswith("#") or line.startswith("\n") or line.startswith("\r"):
            continue
        line = line.replace("\n", "")

        nm = line.split(" ")
        n = int(nm[0])
        m = int(nm[1])
        question = []
        # print(f"N = {n}, M = {m}")

        if n == 0 and m == 0:
            break

        for i in range(m):
            line = file.readline()
            if line.startswith("#") or line.startswith("\n") or line.startswith("\r"):
                continue
            line = line.replace("\n", "")

            lines = [toNumber(c) for c in line]
            # print(line, lines)
            question.append(lines)
        questions.append(question)

    file.close()
    return questions

head = ["File","K??ch th?????c","S??? nh??n" "S??? bi???n", "S??? m???nh ?????", "Th???i gian gi???i"]
data = []
for i in range(1, 5):
    questions = readFile(f"puzzles/inputs{i}")
    with open(f'statistics/inputs{i}.csv', mode='w') as resultFile:
        statisticWriter = csv.writer(resultFile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for j, q in enumerate(questions):
            exe = NumberLink(q)
            try:
                start = time.time()
                exe.solve()
                solve_end = time.time()
                result = [f"input{i}",f"{exe.ROW}x{exe.COL}",exe.Label_count,exe.VAR_COUNT,exe.CLAUSES,round(solve_end-start,3)]
                print(result)
                statisticWriter.writerow(result)
                exe.draw(f"inputs{i}_({exe.ROW}x{exe.COL})_{exe.Label_count}labels_({j})")
                draw_end= time.time()
                data.append(result)
            except Exception as ex:
                data+=[[f"input{i}",f"{exe.ROW}x{exe.COL}",exe.Label_count,exe.VAR_COUNT,exe.CLAUSES,"error"]]
                continue
                # traceback.print_exception(ex)
print(tabulate(data, headers=head, tablefmt="grid"))


# m = [[0, -1, -1, -1, 1, -1, -1, -1, -1]]
# m += [[-1, -1, 3, -1, -1, -1, -1, -1, -1]]
# m += [[-1, -1, -1, -1, -1, -1, -1, -1, -1]]
# m += [[-1, 2, -1, -1, -1, -1, -1, -1, -1]]
# m += [[-1, -1, -1, -1, 2, -1, -1, 0, -1]]
# m += [[1, -1, -1, -1, 4, -1, -1, -1, -1]]
# m += [[-1, -1, -1, -1, -1, -1, -1, -1, 5]]
# m += [[-1, 3, -1, -1, 4, -1, -1, -1, -1]]
# m += [[-1, -1, -1, -1, -1, -1, 5, -1, -1]]

# exe = NumberLink(m)
# exe.solve()
# exe.draw()
