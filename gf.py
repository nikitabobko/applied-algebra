import functools

import numpy as np


def gen_pow_matrix(primpoly):
    q = len(bin(primpoly)) - 3  # primpoly power
    ret = np.zeros((2 ** q - 1, 2)).astype(int)
    max = 0b1 << q  # alpha^q

    normalized = primpoly & (max - 1)  # normalized form of alpha^q
    cur = 0b10
    for i in range(1, 2 ** q):
        ret[i - 1, 1] = cur
        ret[cur - 1, 0] = i
        # print(ret[i - 1, 0], bin(ret[i - 1, 1])) todo
        cur = cur << 1
        if (cur >> q) != 0:
            cur ^= max
            cur ^= normalized
    return ret


def add(X, Y):
    return np.bitwise_xor(X, Y)


def sum(X, axis=0):
    # shape = list(X.shape)
    # shape[0] = 1
    # shape = tuple(shape)
    return np.bitwise_xor.reduce(X, axis=axis)


def prod(X, Y, pm):
    def mult(x, y):
        if x == 0 or y == 0:
            return 0
        pow = (pm[x - 1, 0] + pm[y - 1, 0]) % len(pm)
        # if pow == 0: todo
        #     return 1
        return pm[pow - 1, 1]

    return np.vectorize(mult)(X, Y)


def divide(X, Y, pm):
    def div(x, y):
        assert y != 0
        if x == 0:
            return 0
        pow = (pm[x - 1, 0] - pm[y - 1, 0]) % len(pm)
        # if pow == 0: todo
        #     return 1
        return pm[pow - 1, 1]

    return np.vectorize(div)(X, Y)


def swap(arr, i, j):
    t = np.array(arr[i])
    arr[i] = arr[j]
    arr[j] = t


def linsolve(A, b, pm):
    A = np.asarray(A)
    b = np.asarray(b)
    cols_num = len(A[0])
    rows_num = len(A)
    assert rows_num == cols_num
    for diagonal_index in range(0, cols_num):
        index_ = A[diagonal_index:, diagonal_index]
        nonzero = np.nonzero(index_)
        first_non_zero_index_in_col = nonzero[0]
        if len(first_non_zero_index_in_col) == 0:
            return np.nan
        first_non_zero_index_in_col = first_non_zero_index_in_col[0] + diagonal_index
        swap(A, diagonal_index, first_non_zero_index_in_col)
        swap(b, diagonal_index, first_non_zero_index_in_col)
        cur_first_elem = A[diagonal_index, diagonal_index]
        cur_row = A[diagonal_index]
        cur_b = b[diagonal_index]
        for row_index in range(diagonal_index + 1, rows_num):
            coeff = int(divide(A[row_index, diagonal_index], cur_first_elem, pm))
            A[row_index] = add(A[row_index], prod(np.full(cols_num, coeff), cur_row, pm))
            b[row_index] = int(add(b[row_index], int(prod(coeff, cur_b, pm))))
    ans = np.zeros(cols_num).astype(int)
    ans[rows_num - 1] = int(divide(b[cols_num - 1], A[rows_num - 1, cols_num - 1], pm))
    for diagonal_index in range(rows_num - 2, -1, -1):
        row = A[diagonal_index]
        prev_sum = sum(prod(row[diagonal_index + 1:cols_num], ans[diagonal_index + 1:cols_num], pm))
        cur_ans = int(divide(int(add(b[diagonal_index], prev_sum)), row[diagonal_index], pm))
        ans[diagonal_index] = cur_ans
    return ans

def minpoly(x, pm):
    x = set(x)
    old = set(x)
    for y in old:
        cur = int(prod(y, y, pm))
        while cur != y:
            x.add(cur)
            cur = int(prod(cur, cur, pm))
    min_polynomial = functools.reduce(lambda first, second: polyprod(first, second, pm),
                                      map(lambda it: np.array([1, it]), x))
    return min_polynomial, np.array(list(x))


def normalize_poly(p):
    first_non_zero_index = 0
    for i in range(0, len(p)):
        if p[i] != 0 or i == len(p) - 1:
            first_non_zero_index = i
            break
    ret = p[first_non_zero_index:]
    return ret if len(ret) > 0 else np.array([0])


def normalized_polydeg(p):
    return len(p) - 1


def polydeg(p):
    return normalized_polydeg(normalize_poly(p))


def polyval(p, x, pm):
    p = normalize_poly(p)

    def val(x_elem):
        cur = 0b1
        x_values = np.zeros(len(p)).astype(int)

        for i in range(len(p) - 1, -1, -1):
            x_values[i] = cur
            cur = int(prod(cur, x_elem, pm))

        return sum(prod(x_values, p, pm))

    return np.array(list(map(val, x)))


def polyprod(p1, p2, pm):
    p1, p2 = normalize_poly(p1), normalize_poly(p2)
    p1_deg, p2_deg = normalized_polydeg(p1), normalized_polydeg(p2)
    ret_deg = p1_deg + p2_deg
    ret = np.zeros(ret_deg + 1).astype(int)
    for i in range(0, len(p1)):
        for j in range(0, len(p2)):
            cur_deg = p1_deg - i + p2_deg - j
            ret[ret_deg - cur_deg] = add(ret[ret_deg - cur_deg], int(prod(p1[i], p2[j], pm)))
    return normalize_poly(ret)


def polydiv(p1, p2, pm):
    p1, p2 = normalize_poly(p1), normalize_poly(p2)
    p1_deg, p2_deg = normalized_polydeg(p1), normalized_polydeg(p2)
    if p1_deg == 0 and p1[0] == 0:
        p1_deg = -1
    if p1_deg < p2_deg:
        return np.array([0]), p1
    div_deg = p1_deg - p2_deg
    div = np.zeros(div_deg + 1).astype(int)
    while p1_deg >= p2_deg:
        coeff = int(divide(p1[0], p2[0], pm))
        p_deg = p1_deg - p2_deg
        div[div_deg - p_deg] = coeff
        p2_multiplied = prod(p2, np.full(len(p2), coeff), pm)
        p2_multiplied = np.concatenate([p2_multiplied, np.zeros(p_deg).astype(int)])

        p1 = polyadd(p1, p2_multiplied)
        p1_deg = normalized_polydeg(p1)
        if p1_deg == 0 and p1[0] == 0:
            p1_deg = -1
    return div, normalize_poly(p1)


def polyadd(p1, p2):
    if len(p1) > len(p2):
        p1, p2 = p2, p1
    p1 = np.concatenate([np.zeros(len(p2) - len(p1)).astype(int), p1])
    ret = add(p1, p2)
    return normalize_poly(ret)


def euclid(p1, p2, pm, max_deg=0):
    p1, p2 = normalize_poly(p1), normalize_poly(p2)
    p1_deg, p2_deg = normalized_polydeg(p1), normalized_polydeg(p2)
    swap_flag = False
    if p2_deg > p1_deg:
        swap_flag = True
        p1, p2 = p2, p1
        p1_deg, p2_deg = p2_deg, p1_deg
    E = [[np.array([1]), np.array([0])], [np.array([0]), np.array([1])]]
    # if p2_deg > max_deg:
    r_deg = p2_deg
    r = p2
    while r_deg >= max_deg:
        q, r = polydiv(p1, p2, pm)
        r_deg = normalized_polydeg(r)
        new_E = [[[], []], [[], []]]
        new_E[0][0] = E[0][1]
        new_E[0][1] = polyadd(E[0][0], polyprod(q, E[0][1], pm))
        new_E[1][0] = E[1][1]
        new_E[1][1] = polyadd(E[1][0], polyprod(q, E[1][1], pm))
        E = new_E
        p1, p2 = p2, r
    if swap_flag:
        return p2, E[1][1], E[0][1]
    else:
        return p2, E[0][1], E[1][1]
