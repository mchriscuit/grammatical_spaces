import numpy as np
from scipy.stats import binom


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

    """========== CLASS METHODS ====================================="""

    @classmethod
    def rv(cls, a, size=None):
        return cls.rng.choice(a, size=size)

    @classmethod
    def pmf(cls, a):
        return 1. / len(a)
