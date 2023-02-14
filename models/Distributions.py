import numpy as np
from itertools import product
from Levenshtein import distance
from scipy.stats import binom, geom
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


class Geometric(Distributions):
    """========== INITIALIZATION =========================================="""

    def __init__(self, n, p):
        self._m = n+1
        self._a = list(range(self._m))
        self._p = np.array([geom.pmf(i, p, loc=-1) for i in self._a])
        self._p = self._p / sum(self._p)

    """========== INSTANCE METHODS ========================================"""

    def rv(self):
        return self.rng.choice(a=self._a, p=self._p)

    def pmf(self, k):
        assert self._m >= k, f"Invalid number of successes; {self._m} !>= {k}"
        return self._p[k]
    

class Distance(Distributions):
    """========== INITIALIZATION =========================================="""

    def __init__(
        self,
        segs: np.ndarray,
        n: int,
        dl: int,
        de: int,
    ):
        ## Segment space
        self._segs = np.array(segs)

        ## Size of the forms
        self._n = n
        self._dl = dl
        self._de = de

        """=============== PRIOR DISTRIBUTION ======================= """
        self._pralns, self._prprbs = self.init_all_alns(self._n, self._dl)
        self._prpmfs = {
            x: {y: p for y, p in zip(Y, self._prprbs[x])}
            for x, Y in self._pralns.items()
        }

        """=============== PROPOSAL DISTRIBUTION ==================== """
        self._tpalns, self._tpprbs = self.init_all_alns(self._n, self._de)
        self._tppmfs = {
            x: {y: p for y, p in zip(Y, self._tpprbs[x])}
            for x, Y in self._tpalns.items()
        }

    """========== INSTANCE METHODS ========================================"""

    def compute_distance(self, d: float, x: str, Y: list):
        """Calculates the exponentiated distance between x and Y"""
        dist = {x: [np.exp(-d * distance(x, y)) for y in Y]}
        return dist

    def init_all_alns(self, n: int, d: float):
        """Initializes the space of possible edits of a pair of strings of
        length (0,0) to (n,n)
        """
        alns = {}
        prbs = {}
        Y = []
        for i in range(n + 1):
            Y += list(map("".join, product(self._segs, repeat=i)))
        for i in tqdm(range(n + 1)):
            X = list(map("".join, product(self._segs, repeat=i)))
            for x in X:
                alns[x] = np.array(Y, dtype=object)
                prbs[x] = [np.exp(-d * distance(x, y)) for y in Y]
                prbs[x] = np.array(prbs[x]) / sum(prbs[x])
        return alns, prbs

    def prrv(self, x: str):
        return self.rng.choice(self.pralns(x), p=self.prprbs(x))

    def tprv(self, x: str):
        return self.rng.choice(self.tpalns(x), p=self.tpprbs(x))

    """========== ACCESSORS ==========================================="""

    def pralns(self, x: str):
        return self._pralns[x]

    def prprbs(self, x: str):
        return self._prprbs[x]

    def tpalns(self, x: str):
        return self._tpalns[x]

    def tpprbs(self, x: str):
        return self._tpprbs[x]

    def prpmf(self, x: str, y: str):
        return self._prpmfs[x][y]

    def tppmf(self, x: str, y: str):
        return self._tppmfs[x][y]
