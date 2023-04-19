import numpy as np
from itertools import product
from weighted_levenshtein import levenshtein

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
    def __init__(self, n: int):
        ## Initialize random variates and probabilities
        self._vs = np.arange(n)
        self._ps = np.full(n, 1 / float(n))

        ## Initialize the superclass
        super().__init__(self._vs, self._ps)


class Kniform(BaseDistribution):
    def __init__(self, n: int, k: int, t: float):
        ## Initialize random variates and probabilities
        self._vs = np.arange(k + 1)
        self._ps = Kniform.p(n, self._vs, t)
        self._ps = self._ps / self._ps.sum()

        ## Initialize the superclass
        super().__init__(self._vs, self._ps)

    @staticmethod
    def p(n: int, k: np.ndarray, t: float):
        return 1 / float(n) ** (k.astype(float) * t)


class Bernoulli(BaseDistribution):
    def __init__(self, p: float):
        ## Initialize random variates and probabilities
        self._vs = np.array([0, 1])
        self._ps = np.array([1 - p, p])

        ## Initialize the superclass
        super().__init__(self._vs, self._ps)


class Keometric(BaseDistribution):
    def __init__(self, n: int, p: float):
        ## Initialize random variates and probabilities
        self._vs = np.arange(n + 1)
        self._ps = Keometric.p(self._vs, p)
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
    def __init__(self, w: np.ndarray, p: np.ndarray):
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
    def __init__(self, ss: np.ndarray, n: int = 1):
        ## Segmental information and costs
        self._ss = ss
        self._ns = ss.size

        ## Base distribution
        self._bn = n + 1
        self._bw = self.it_prod(n + 1)
        self._ic = np.ones(128, dtype=np.float64)
        self._dc = np.ones(128, dtype=np.float64)
        self._sc = np.ones((128, 128), dtype=np.float64)

    @property
    def bn(self):
        return self._bn

    @property
    def bw(self):
        return self._bw

    @property
    def sc(self):
        return self._sc

    @property
    def ic(self):
        return self._ic

    @property
    def dc(self):
        return self._dc

    def add_derv(self, name: str, k: int, c: dict = None):
        """Creates a derived distribution of strings and distances from length 0 to k < n"""

        ## Check that the derived distribution is a subset of the base
        assert k <= self._bn, f"invalid value for k: {k} !<= {self._bn}"

        ## Save the length of the derived distribution
        setattr(self, name + "_dk", k)

        ## Get the view of the string distributions up to length k
        dw = self.it_prod(n=k)
        setattr(self, name + "_dw", dw)

        ## Calculate the edit distance between all points in the matrix
        dic = np.ones(128, dtype=np.float64)
        ddc = np.ones(128, dtype=np.float64)
        dd, dsc = self.it_dist(w=dw, c=c)
        setattr(self, name + "_dd", dd)
        setattr(self, name + "_dic", dic)
        setattr(self, name + "_ddc", ddc)
        setattr(self, name + "_dsc", dsc)

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
        dp = dp / (dp.sum(axis=1)[:, None]) ** int(nrm)

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

    def it_dist(self, w: np.ndarray, c: dict = None):
        """Takes in a one-dimensional array of strings and returns a matrix of
        (symmetric) Levenshtein distances between each pair of strings"""

        ## If a custom operation cost is specified, build to cost matrix
        ## The API takes in a 128 x 128 matrix corresponding to each ASCII
        ## character, where each cell corresponds to the cost. For now, we assume
        ## that only substitutions are non-one
        sc = np.ones((128, 128), dtype=np.float64)
        if c is not None:
            for ss, sm in c.items():
                sc[ord(ss[0]), ord(ss[1])] = 1 - sm

        ## Calculate the symmetric distances between each pair of strings
        l = w.tolist()
        d = np.asarray([[levenshtein(x, y, substitute_costs=sc) for y in l] for x in l])

        return d, sc
