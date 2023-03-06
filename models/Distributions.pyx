## Basic Python package imports
from Levenshtein import distance
from tqdm import tqdm

## Basic Cython package imports
import numpy as np
cimport numpy as np

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                DISTRIBUTIONS DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


cdef class Distributions:

    """========== CLASS METHODS ====================================="""

    rng = np.random.default_rng()
    factorial = np.math.factorial


class Binomial(Distributions):

    """========== CLASS METHODS ====================================="""

    @classmethod
    def rv(cls, unsigned short n, float p):
        return cls.rng.binomial(n, p)

    @classmethod
    def pmf(cls, unsigned short k, unsigned short n, float p):
        assert n >= k, f"Invalid number of successes; {n} !>= {k}"
        c = cls.factorial(n) / (cls.factorial(k) * cls.factorial(n - k))
        return c * (p**k) * (1 - p) ** (n - k)


class Bernoulli(Distributions):

    """========== CLASS METHODS ====================================="""

    @classmethod
    def rv(cls, float p):
        return cls.rng.binomial(n=1, p=p)

    @classmethod
    def pmf(cls, unsigned short k, float p):
        return p if k else 1 - p


class Uniform(Distributions):

    """========== CLASS METHODS ==========================================="""

    @classmethod
    def rv(cls, unsigned short n):
        return cls.rng.choice(n)

    @classmethod
    def pmf(cls, unsigned short n):
        return 1.0 / float(n)


class Geometric(Distributions):

    """========== CLASS METHODS ==========================================="""
    
    @classmethod
    def cpmf(cls, unsigned short k, float p):
        return (p) * (1.0 - p) ** (k)

    """========== INITIALIZATION =========================================="""

    def __init__(self, unsigned short n, float p):
        self._m = n+1
        self._a = list(range(self._m))
        self._p = np.array([Geometric.cpmf(i, p) for i in self._a])
        self._p = self._p / sum(self._p)

    """========== INSTANCE METHODS ========================================"""

    def rv(self):
        return self.rng.choice(a=self._a, p=self._p)

    def pmf(self, unsigned short k):
        assert self._m >= k, f"Invalid number of successes; {self._m} !>= {k}"
        return self._p[k]
    

cdef class Distance(Distributions):
    """========== CLASS ATTRIBUTES ========================================"""

    ## Segment inventory
    cdef np.ndarray _segs

    ## Parameters over the edit space
    cdef unsigned short _n
    cdef float _dl
    cdef float _de

    ## Prior and proposal distributions
    cdef dict _pralns, _prprbs, _prpmfs
    cdef dict _tpalns, _tpprbs, _tppmfs

    """========== INITIALIZATION =========================================="""

    def __init__(self, np.ndarray segs, unsigned short n, float dl, float de):

        ## Segment inventory
        self._segs = np.array(segs)

        ## Parameters over the edit space
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

    def compute_distance(self, float d, str x, list Y):
        """Calculates the exponentiated distance between x and Y"""
        dist = {x: np.exp([-d * distance(x, y) for y in Y])}
        return dist

    def init_all_alns(self, unsigned short n, float d):
        """Initializes the space of possible edits of a pair of strings of
        length (0,0) to (n,n)
        """
        def product(np.ndarray segments, unsigned short repeat):

            ## Generate the segments list, repeated ``length`` times
            rsegments = [list(segments)] * repeat

            ## For each list, compute the possible permutations
            ## for the given length
            osegments = [[]]
            for rsegment in rsegments:
                osegments = [osegment+[r] for osegment in osegments for r in rsegment]

            ## Join the tuples into a string
            osegments = ["".join(osegment) for osegment in osegments]
            return osegments
  
        ## Initialize counters
        cdef unsigned short i
        cdef str x
        cdef str y

        ## Initialize object variables
        cdef dict alns = {}
        cdef dict prbs = {}
        cdef list X = []
        cdef list Y = []

        ## Loop through and generate all possible URs of length 0 to N
        for i in range(n + 1):
            Y += product(self._segs, repeat=i)

        ## Loop through and calculate the probabilities getting from
        ## all possible UR strings to another
        for i in tqdm(range(n + 1)):
            X = product(self._segs, repeat=i)
            for x in X:
                alns[x] = np.array(Y, dtype=object)
                prbs[x] = np.exp([-d * distance(x, y) for y in Y])
                prbs[x] = np.array(prbs[x]) / sum(prbs[x])
        
        return alns, prbs

    def prrv(self, str x):
        return self.rng.choice(self.pralns(x), p=self.prprbs(x))

    def tprv(self, str x):
        return self.rng.choice(self.tpalns(x), p=self.tpprbs(x))

    """========== ACCESSORS ==========================================="""

    def pralns(self, str x):
        return self._pralns[x]

    def prprbs(self, str x):
        return self._prprbs[x]

    def tpalns(self, str x):
        return self._tpalns[x]

    def tpprbs(self, str x):
        return self._tpprbs[x]

    def prpmf(self, str x, str y):
        return self._prpmfs[x][y]

    def tppmf(self, str x, str y):
        return self._tppmfs[x][y]
