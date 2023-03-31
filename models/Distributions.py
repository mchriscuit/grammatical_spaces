import numpy as np

from tqdm import tqdm
from itertools import product
from polyleven import levenshtein


class Distributions:
    rng = np.random.default_rng()


class Uniform(Distributions):

    """========== CLASS METHODS ===================================================="""

    @classmethod
    def rv(cls, n: int):
        return cls.rng.choice(n)

    @classmethod
    def pmf(cls, n: int):
        return 1.0 / float(n)


class Bernoulli(Distributions):

    """========== CLASS METHODS ===================================================="""

    @classmethod
    def rv(cls, p: float, size: int):
        return cls.rng.binomial(n=1, p=p, size=size)

    @classmethod
    def pmf(cls, k: np.ndarray, p: float):
        return (p) ** (k) * (1.0 - p) ** (1 - k)


class Geometric(Distributions):

    """========== CLASS METHODS ===================================================="""

    @classmethod
    def cpmf(cls, k: int, p: float):
        return (p) * (1.0 - p) ** (k)

    """========== INITIALIZATION ==================================================="""

    def __init__(self, n: int, p: float):
        self._m = n + 1
        self._a = np.arange(self._m)
        self._p = np.array([Geometric.cpmf(i, p) for i in self._a])
        self._p = self._p / sum(self._p)

    """========== INSTANCE METHODS ================================================="""

    def rv(self, size: int):
        return self.rng.choice(a=self._a, p=self._p, size=size)

    def pmf(self, k: np.ndarray):
        return self._p[k]


class Distance(Distributions):

    """========== INITIALIZATION ==================================================="""

    def __init__(
        self, segs: np.ndarray, n: int, e: int, nc: float, dl: float, de: float
    ):

        ## Segment inventory
        self._segs = np.array(segs)

        ## Parameters over the edit space
        self._m = n + 1
        self._e = e
        self._b = 5
        self._nc = nc
        self._dl = dl
        self._de = de

        ## Alignments and distances
        self._O, N = self.init_dists(self._m + self._b, self._e)
        self._I, P = self.init_dists(self._m, self._e)

        """=============== INPUTS AND OUTPUTS ======================= """
        self._ncprbs = np.exp(-N * self._nc)

        """=============== PRIOR DISTRIBUTION ======================= """
        self._prprbs = np.exp(-P * self._dl)
        self._prprbs = self._prprbs / self._prprbs.sum(axis=1)[:, None]

        """=============== PROPOSAL DISTRIBUTION ==================== """
        self._tpprbs = np.exp(-P * self._de)
        self._tpprbs = self._tpprbs / self._tpprbs.sum(axis=1)[:, None]

    """========== INSTANCE METHODS ================================================"""

    def init_dists(self, m: int, e: int):
        """Initializes the space of possible edits of a pair of strings of
        length (0,0) to (n,n)
        """

        ## Loop through and generate all possible URs of length 0 to N
        A = []
        for i in range(m):
            A += ["".join(a) for a in product(self._segs, repeat=i)]

        ## Calculate the distances between each sorted string combination
        A = sorted(A)
        D = [[levenshtein(a, x) for x in A] for a in tqdm(A)]

        ## Convert to numpy arrays
        A = np.array(A)
        D = np.array(D)

        return A, D

    """========== ACCESSORS ========================================================"""

    @property
    def I(self):
        return self._I

    @property
    def O(self):
        return self._O

    @property
    def ncpmf(self):
        return self._ncprbs

    def prpmf(self, idx: tuple):
        return self._prprbs[idx]

    def tppmf(self, idx: tuple):
        return self._tpprbs[idx]

    def rv(self, p: np.ndarray, axis: int = 1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)
