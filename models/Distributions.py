import numpy as np
from itertools import product, combinations
from polyleven import levenshtein

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
                SUPERCLASS DECLARATION
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class BaseDistribution:
    def __init__(self, vs: np.ndarray, ps: np.ndarray):
        self._vs = vs
        self._ps = ps
        self.rng = np.random.default_rng()

    def rvs(self, size: int = None):
        return self.rng.choice(a=self._vs, p=self._ps, size=size)

    def pmf(self, k: np.ndarray):
        return self._ps[k]


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
                SUBCLASS DISTRIBUTIONS
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Uniform(BaseDistribution):
    def __init__(self, n: float):

        ## Initialize random variates and probabilities
        self._vs = np.arange(n)
        self._ps = np.full(n, 1 / n)

        ## Initialize the superclass
        super().__init__(self._vs, self._ps)


class Bernoulli(BaseDistribution):
    def __init__(self, p: float):

        ## Initialize random variates and probabilities
        self._vs = np.array([0, 1])
        self._ps = np.array([1 - p, p])

        ## Initialize the superclass
        super().__init__(self._vs, self._ps)


class Geometric(BaseDistribution):
    def __init__(self, n: int, p: float):

        ## Initialize random variates and probabilities
        self._vs = np.arange(n + 1)
        self._ps = Geometric.p(self._vs, p)
        self._ps = self._ps / self._ps.sum()

        ## Initialize the superclass
        super().__init__(self._vs, self._ps)

    @staticmethod
    def p(k: np.ndarray, p: float):
        return p * (1.0 - p) ** k


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
                CUSTOM DISTANCE DISTRIBUTION
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class CustomDistribution:
    def __init__(self, w: np.ndarray, p: float):
        self._w = w
        self._p = p

    def rvs(self, idx, axis=1):
        p = self._p[idx]
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def pmf(self, idx):
        return self._p[idx]

    @property
    def w(self):
        return self._w

    @property
    def p(self):
        return self._p


class Distance:
    def __init__(self, ss: np.ndarray, cs: dict = {}, n: int = 1):

        ## Segmental information and costs
        self._ss = ss
        self._ns = ss.size
        self._cs = cs

        ## Base distribution
        self._bn = n + 1
        self._bw = self.it_prod(n + 1)

    @property
    def bn(self):
        return self._bn

    @property
    def bw(self):
        return self._bw

    def add_derv(self, name: str, k: int):
        """Creates a derived distribution of strings and distances from length 0 to k < n"""

        ## Check that the derived distribution is a subset of the base
        assert k <= self._bn, f"invalid value for k: {k} !<= {self._bn}"

        ## Save the length of the derived distribution
        setattr(self, name + "_dk", k)

        ## Get the view of the string distributions up to length k
        dw = self.it_prod(k)
        setattr(self, name + "_dw", dw)

        ## Calculate the edit distance between all points in the matrix
        dd = self.it_dist(dw)
        setattr(self, name + "_dd", dd)

    def add_pdis(self, pname: str, dname: str, c: float, nrm: bool = True):
        """Creates a custom probability distribution given an initialized derived
        string distribution, distance, and cost value. The name of the pdist must
        be the name of a derived distribution. Returns a normalized value if set as True
        """

        ## Check that the name exists as a derived distribution
        assert hasattr(self, dname + "_dw"), "derived distribution not initialized"

        ## Retrieve the information of the derived distribution
        dw = getattr(self, dname + "_dw")
        dp = getattr(self, dname + "_dd").copy().astype(float)

        ## Calculate the exponentiated distances multiplied by the cost value
        dp = np.exp(-c * dp, dp)

        ## Normalize the distances, if set as True
        dp = dp / (dp.sum(axis=1)[:None]) ** int(nrm)

        ## Create a CustomDistribution object
        setattr(self, pname, CustomDistribution(dw, dp))

    def it_prod(self, n: int):
        """Computes the (sorted) space of all possible string permutations
        from length 0 to length n
        """

        ## Retrieve variables
        t = range(n + 1)
        s = self._ss

        ## Generate (sub) products, save to numpy array and sort for future access
        w = np.fromiter(
            ("".join(x) for i in t for x in product(s, repeat=i)), dtype="U" + f"{n+1}"
        )
        w.sort()

        return w

    def it_dist(self, w: np.ndarray):
        """Takes in a one-dimensional array of strings and returns a matrix of
        (symmetric) Levenshtein distances between each pair of strings"""

        ## Calculate the symmetric distances between each pair of strings
        l = w.tolist()
        d = np.asarray([[levenshtein(x, y) for y in l] for x in l])

        return d
