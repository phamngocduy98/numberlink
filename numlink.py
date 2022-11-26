import time
import matplotlib.pyplot as plt
import random
from pysat.solvers import Solver
from sat import BinaryEncoding, BinomialEncoding


# Sinh tổ hợp chập k của N
memo = {}


def C(k, N):
    key = f"{k}-{N}"
    if (key in memo):
        return memo[key]
    # print(f"C{k}:{N}")
    result = []
    used = [False]*N
    temp = [0]*k

    def backtrack(i):
        if (i == k):
            result.append(temp.copy())
            return
        for j in range(N):
            if (not used[j]):
                if (i > 0 and j < temp[i-1]):
                    continue
                temp[i] = j
                used[j] = True
                backtrack(i+1)
                used[j] = False
    backtrack(0)
    memo[key] = result.copy()
    return result.copy()

# Binomal


duybinomial = BinomialEncoding(0)


def at_least_binomal(vars, k, condition=True):
    return duybinomial.atLeastK(k, vars)

    encoding_data = []
    for com in C(len(vars)-k+1, len(vars)):
        clause = []
        for idx in com:
            clause.append(vars[idx] if condition else -vars[idx])
        encoding_data.append(clause.copy())
    return encoding_data.copy()


def at_most_binomal(vars, k, condition=True):
    return duybinomial.atMostK(k, vars)
    return at_least_binomal(vars, len(vars)-k, not condition).copy()


def exactly_binomal(total_var, vars, k, condition=True):
    return duybinomial.exactK(k, vars), 0
    return (
        at_most_binomal(vars, k, condition) +
        at_least_binomal(vars, k, condition),
        0  # add_var_count
    )


# cnf, varcount = exactly_binomal(4, [1, 2, 3, 4, 5], 2, True)
# print(cnf)
# cnf+=[[-1]]
# cnf+=[[-2]]
# cnf+=[[-3]]

# sat_solve(cnf)


class NumberLink:
    def __init__(self, input, exact_encoder=exactly_binomal):
        self.input = input
        self.ROW = len(input)
        self.COL = len(input[0])
        self.exact_encoder = exact_encoder
        self.Label_count = 0
        self.Labels = []
        for row in range(self.ROW):
            for col in range(self.COL):
                if (input[row][col] >= 0 and input[row][col] not in self.Labels):
                    self.Labels.append(input[row][col])
        self.Label_count = len(self.Labels)
        # print("Label", self.Labels)
        self.MAX_VALUE = self.VAR_COUNT = self.ROW * self.COL * self.Label_count
        self.colors = ["#"+''.join([random.choice('23456789ABCDEF') for j in range(6)])
                       for i in range(self.Label_count)]

    def _draw(self, input):
        if (len(input)):
            figSizeW = 0.7*self.COL
            figSizeH = 0.7*self.ROW
            cellSize = 0.7/figSizeW
            fig, ax1 = plt.subplots(figsize=(figSizeW, figSizeH))
            ax1.axis('off')

            table = ax1.table(
                cellColours=[["0"], ["0"]], loc='center', cellLoc='center')
            for row in range(self.ROW):
                for col in range(self.COL):
                    num = input[row][col]
                    label = self.Labels[num] if num != -1 else -2
                    table.add_cell(row=row, col=col, width=cellSize, height=cellSize,
                                   text=f"{ label+ 1}" if label >= 0 else "", loc="center", facecolor=self.colors[num] if num >= 0 else "w")
            table.set_fontsize(12)

            plt.savefig("numberlink.png", dpi=200, bbox_inches='tight')

    def draw(self):
        # self._draw(self.input)
        self._draw(self.result)

    def get_index(self, i, j, k):
        return self.Label_count*(self.COL*i + j) + k + 1

    def get_loc(self, idx):
        idx = idx - 1
        k = idx % self.Label_count
        idx = idx // self.Label_count
        j = idx % self.COL
        i = idx // self.COL
        return (i, j, k)

    def get_number_link_matrix(self, sat_solve_result):
        number_link_matrix = [[-1]*self.COL for i in range(self.ROW)]
        for var in sat_solve_result:
            if (self.idx_in_matrix(var)):
                i, j, k = self.get_loc(var)
                number_link_matrix[i][j] = k
        return number_link_matrix

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
        solver = Solver(name="minisat22",use_timer=True, bootstrap_with=[])
        binomial = BinomialEncoding(1)
        binary = BinaryEncoding(self.VAR_COUNT)
        self.startTime = time.time()
        for row in range(self.ROW):
            for col in range(self.COL):
                labels = [self.get_index(row, col, self.Labels[num])
                          for num in range(self.Label_count)]
                rules = binomial.atMostOne(labels)  # Chính xác 1 nhãn được chọn cho 1 ô
                solver.append_formula(rules)

                if (self.input[row][col] >= 0):  # Nhãn được điền sẵn
                    label = self.get_index(row, col, self.input[row][col])
                    solver.add_clause([label])

                    # Chính xác 1 ô trong 4 ô xung quanh được điền nhãn giống nó
                    vars = self.list_neighbor_cell_index(
                        row, col, self.input[row][col])
                    rules= binomial.exactOne(vars)
                    solver.append_formula(rules)
                else:
                    for num in range(self.Label_count):
                        label = self.get_index(
                            row, col, self.Labels[num])  # Nhãn giả định
                        # Lấy danh sách biến các ô xung quanh với nhãn giống nhãn giả định
                        nbr_label_vars = self.list_neighbor_cell_index(
                            row, col, self.Labels[num])

                        # Nếu nhãn giả định được chọn -> chính xác 2 ô xung quanh được gán nhãn giống nó
                        # Chính xác 2 ô xung quanh có nhãn giống nhãn giả định // Luật tạm theo nhãn giả định
                        rules_cnf = binomial.exactK(2, nbr_label_vars)
                        # label -> rules_cnf  <=> -label V rules <=> -label V (rule1 ^ rule2 ^ ....) <=> (-label V rule1) ^ (-label V rule2) .....
                        for rule in rules_cnf:
                            solver.append_formula([[-label] + rule])
        self.buildTime = time.time()
        solver.solve()
        self.accTime = solver.time_accum()
        result = solver.get_model()
        self.satTime = time.time()
        self.result = self.get_number_link_matrix(result) if result else []
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

print("input,   n,   m, varcount, labelcount, buildClauseTime, solveTime, totalTime")
for i in range(6, 11):
    questions = readFile(f"puzzles/inputs{i}")
    for q in questions:
        try:
            exe = NumberLink(q)
            exe.solve()
            print(f"{i:5}, {exe.COL:3}, {exe.ROW:3}, {exe.VAR_COUNT:8}, {exe.Label_count:10}, {exe.getBuildTime():15.2f}, {exe.getSATTime():9.2f}, {exe.getTotalTime():9.2f}")
        except Exception as ex:
            print("EX",ex)

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
