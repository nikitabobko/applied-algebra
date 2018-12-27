import math
import gf
import numpy as np


class BCH:
    primopolys = [-1, -1, 7, 11, 19, 55, 109, 193, 425, 617, 1849, 3085, 7057, 14909, 28199, 41993, 112987]
    pm = None
    g = None
    R = None

    def __init__(self, n, t):
        q = int(math.log2(n + 1))
        assert 2 <= q <= 16
        assert 2 ** q - 1 == n  # todo
        assert 2 * t + 1 <= n
        self.pm = gf.gen_pow_matrix(self.primopolys[q])
        self.R = self.pm[0:(2 * t), 1]
        self.g, _ = gf.minpoly(self.R, self.pm)
        assert np.logical_or((self.g == [0]), (self.g == [1])).all()
        x = np.zeros(n + 1).astype(int)
        x[0] = 1
        x[-1] = 1
        assert (gf.polydiv(x, self.g, self.pm)[1] == [0]).all()

    def encode(self, U):
        def encode_elem(it):
            n = len(self.pm)
            m = gf.normalized_polydeg(self.g)
            assert n - m == len(it)
            x_pow_m = np.zeros(m + 1).astype(int)
            x_pow_m[0] = 1
            a = gf.polyprod(it, x_pow_m, self.pm)
            _, mod = gf.polydiv(a, self.g, self.pm)
            n = len(self.pm)
            ret = gf.polyadd(a, mod)
            ret = np.concatenate([np.zeros(n - len(ret)).astype(int), ret])

            assert (gf.polydiv(ret, self.g, self.pm)[1] == [0]).all()
            assert (gf.polyval(ret, self.R, self.pm) == [0]).all()

            return ret

        return np.array(list(map(encode_elem, U)))

    def decode(self, W, method='euclid'):
        assert method == 'euclid' or method == 'pgz'

        global object_flag
        object_flag = False

        def decode_elem(it):
            global object_flag
            n = len(it)
            assert n == len(self.pm)
            t = len(self.R) // 2
            m = gf.polydeg(self.g)
            s = gf.polyval(it, self.R, self.pm)
            if (s == 0).all():
                return it
            if method == 'pgz':
                lambda_ = np.nan
                for nu in range(t, 0, -1):
                    A = [[s[j] for j in range(0 + i, nu + i)] for i in range(0, nu)]
                    b = [s[i] for i in range(nu, 2 * nu)]
                    lambda_ = gf.linsolve(A, b, self.pm)
                    if lambda_ is not np.nan:
                        break
                if lambda_ is np.nan:
                    object_flag = True
                    return np.full(n, np.nan)
                lambda_ = np.concatenate([lambda_, [1]])
            else:
                z = np.zeros(2 * t + 2).astype(int)
                z[0] = 1
                S = np.concatenate([s[::-1], [1]])
                _, _, lambda_ = gf.euclid(z, S, self.pm, max_deg=t)
            possible_roots = gf.polyval(lambda_, self.pm[:, 1], self.pm)
            roots_count = 0
            for i in range(0, len(possible_roots)):
                if possible_roots[i] == 0:
                    position = self.pm[gf.divide(1, self.pm[i, 1], self.pm) - 1, 0]
                    it[n - position - 1] = 1 - it[n - position - 1]
                    roots_count += 1
            if roots_count != gf.polydeg(lambda_):
                object_flag = True
                return np.full(n, np.nan)
            return it

        ret = list(map(decode_elem, W))
        if object_flag:
            return np.array(ret, dtype=object)
        else:
            return np.array(ret)

    def dist(self):
        m = gf.polydeg(self.g)
        n = len(self.pm)
        k = n - m
        cur = 1
        last = 2 ** k - 1
        msgs = []
        while cur <= last:
            msg = []
            copy = cur
            while copy != 0:
                msg.insert(0, copy & 1)
                copy >>= 1
            msg = np.concatenate([np.zeros(k - len(msg)).astype(int), msg])
            msgs += [msg]
            cur += 1
        coded = self.encode(msgs)
        return np.min(np.count_nonzero(coded, axis=1))
