import numpy as np
from itertools import product
from Levenshtein import distance
from scipy.stats import binom
from tqdm import tqdm


class Distributions:

    """========== CLASS METHODS ====================================="""

    rng = np.random.default_rng()
    factorial = np.math.factorial


class Binomial(Distributions):

    """========== CLASS METHODS ====================================="""

    @classmethod
    def rv(cls, n, p, size=None):
        return cls.rng.binomial(n, p, size=size)

    @classmethod
    def pmf(cls, k, n, p):
        assert n >= k, f"Invalid number of successes; {n} !>= {k}"
        c = cls.factorial(n) / (cls.factorial(k) * cls.factorial(n - k))
        return c * (p**k) * (1 - p) ** (n - k)

    @classmethod
    def trunc_pmf(cls, b, k, n, p):
        assert k >= b, f"Invalid lower bound; {k} !>= {b}"
        assert n >= k, f"Invalid number of successes; {n} !>= {k}"
        if b == 0:
            return cls.pmf(k, n, p)
        else:
            return cls.pmf(k, n, p) / (cls.pmf(b, n, p) + binom.sf(b, n, p))


class Bernoulli(Distributions):

    """========== CLASS METHODS ====================================="""

    @classmethod
    def rv(cls, p, size=None):
        return cls.rng.binomial(n=1, p=p, size=size)

    @classmethod
    def pmf(cls, k, p):
        return p if k else 1 - p


class Uniform(Distributions):

    """========== CLASS METHODS ==========================================="""

    @classmethod
    def rv(cls, a, size=None):
        return cls.rng.choice(a, size=size)

    @classmethod
    def pmf(cls, n):
        return 1.0 / n


class Distance(Distributions):
    """========== INITIALIZATION =========================================="""

    def __init__(
        self,
        segs: np.ndarray,
        costs: dict,
        n: int,
        k: int,
        d: int,
    ):
        self._n = n
        self._k = k
        self._d = d
        self._segs = np.array(segs)
        self._costs = costs

        """=============== LENGTH-BASED EDITS (PRIOR) ================ """
        self._lalns = {}
        self._lprbs = {}
        self.init_all_lalns(self._n)
        self._lidxs = {
            x: {y: i for i, y in enumerate(Y)} for x, Y in self._lalns.items()
        }

        """=============== EDIT-BASED EDITS (TRANSITION) ============ """
        self._ealns = {}
        self._eprbs = {}
        self.init_all_ealns(self._n, self._k)
        self._eidxs = {
            x: {y: i for i, y in enumerate(Y)} for x, Y in self._ealns.items()
        }

    """========== INSTANCE METHODS ========================================"""

    def init_all_lalns(self, n: int):
        """Initializes the space of possible edits of a pair of strings of
        length (0,0) to (n,n)
        """
        Y = []
        for i in range(n + 1):
            Y += list(map("".join, product(self._segs, repeat=i)))
        for i in tqdm(range(n + 1)):
            X = list(map("".join, product(self._segs, repeat=i)))
            for x in X:
                self._lalns[x] = np.array(Y, dtype=object)
                self._lprbs[x] = [np.exp(-self._d * distance(x, y)) for y in Y]
                self._lprbs[x] = np.array(self._lprbs[x]) / sum(self._lprbs[x])

    def init_all_ealns(self, n: int, k: int):
        """Generates all possible alignments from an input string up to length
        n that are at most k edits away. If the resulting alignment is greater
        than n, remove it
        """

        def gen_ealns(a: str):
            """Generates all possible alignments from input string of length i
            that are 1 edit distance away
            """
            sp = [(a[:k], a[k:]) for k in range(len(a) + 1)]
            ir = [l + s + r for l, r in sp for s in self._segs]
            dl = [l + r[1:] for l, r in sp if r]
            sb = [l + s + r[1:] for l, r in sp if r for s in self._segs]
            return set(ir + dl + sb)

        for i in tqdm(range(n + 1)):
            X = list(map("".join, product(self._segs, repeat=i)))
            A = set(X)
            Y = set()
            for j in range(k):
                A = [gen_ealns(a) for a in list(A)]
                A = set.union(*A)
                Y = Y.union(A)
            Y = [y for y in sorted(Y) if len(y) < self._n]
            for x in X:
                self._ealns[x] = np.array(Y, dtype=object)
                self._eprbs[x] = [np.exp(-self._d * distance(x, y)) for y in Y]
                self._eprbs[x] = np.array(self._eprbs[x]) / sum(self._eprbs[x])

    def lrv(self, x: str):
        return self.rng.choice(self.lalns(x), p=self.lprbs(x))

    def erv(self, x: str):
        return self.rng.choice(self.ealns(x), p=self.eprbs(x))

    """========== ACCESSORS ==========================================="""

    def lalns(self, x: str):
        return self._lalns[x]

    def lprbs(self, x: str):
        return self._lprbs[x]

    def ealns(self, x: str):
        return self._ealns[x]

    def eprbs(self, x: str):
        return self._eprbs[x]

    def lpmf(self, x: str, y: str):
        id = self._lidxs[x][y]
        return self._lprbs[x][id]

    def epmf(self, x: str, y: str):
        id = self._eidxs[x][y]
        return self._eprbs[x][id]
