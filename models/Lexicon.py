import numpy as np
import re
from copy import copy, deepcopy
from optim.Inventory import Inventory
from optim.Distributions import Binomial, Bernoulli, Uniform, Distance

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                LEXICON DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Lexicon(Inventory):
    """========== INITIALIZATION ======================================="""

    def __init__(
        self,
        lxs: list,
        clxs: list,
        ur_params: dict,
        tokens: list,
        feats: list,
        configs: list,
    ):

        self._rng = np.random.default_rng()

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(tokens, feats, configs)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._n = ur_params["n"]
        self._k = ur_params["k"]
        self._tht = ur_params["tht"]
        self._psi = ur_params["psi"]
        self._cst = ur_params["costs"]
        self.D = Distance(self.segs(), self._cst, self._n, self._k, self._psi)

        ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs = lxs
        self._clxs = clxs

        ## *=*=*= LEXICAL DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        """Lexeme to lexical sequence"""
        self._lx2clxs = {x: cxs for x, cxs in zip(self._lxs, self._clxs)}
        self._lx2clx2id = {
            x: {cx: i for i, cx in enumerate(self._lx2clxs[x])} for x in self._lxs
        }
        """Lexical sequence to lexeme"""
        self._clx2lxs = {
            cx: [x for x in self._lxs if x in cx] for cxs in self._clxs for cx in cxs
        }
        self._clx2lx2id = {
            cx: {x: i for i, x in enumerate(self._clx2lxs[cx])}
            for cx in self._clx2lxs.keys()
        }

        ## *=*=*= LEXICON HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._lx2ur = {}
        self._lx2pr = {}

    """ ========== OVERLOADING METHODS =================================== """

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "lx2ur" or k == "lx2pr":
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, copy(v))
        return result

    """ ========== INSTANCE METHODS ===================================== """

    def initialize_urs(self, init_ur_hyp):
        """Initializes the UR hypotheses for the lexemes"""
        init_lxs, init_clxs, init_urs = init_ur_hyp
        assert set(self._lxs) == set(
            init_lxs
        ), "Given lexemes and observed lexemes are not equal"
        assert all(
            set(self._lx2clxs[lx]) == set(init_clxs[i]) for i, lx in enumerate(init_lxs)
        ), "Given lexical contexts and observed lexical contexts not equal"
        for init_lx, init_ur in zip(init_lxs, init_urs):
            self._lx2ur[init_lx] = init_ur
            self._lx2pr[init_lx] = self.calculate_pr(init_ur)

    def set_ur(self, lx, ur, pr):
        """Set the given lexeme to the given UR"""
        self._lx2ur[lx] = ur
        self._lx2pr[lx] = pr

    def sample_ur(self, lx, ur_old, inplace=False):
        """Samples a UR for a given lexeme given the proposal distribution.
        Also returns the transition probability and prior of the proposed UR
        """
        for i, u in enumerate(ur_old):
            if i == 0:
                pro_ur = self.D.erv(u)
                ur_new = [pro_ur]
                pro_len = len(pro_ur)
                pr = Binomial.pmf(pro_len, self._n, self._tht)
                pr *= Uniform.pmf(self.nsegs()) ** pro_len
            else:
                cxt_ur = self.D.erv(u)
                ur_new.append(cxt_ur)
                pr *= self.D.lpmf(pro_ur, cxt_ur)
        if inplace:
            self.set_ur(lx, ur_new, pr)
        return ur_new, pr

    def calculate_pr(self, ur):
        """Returns the prior probability of the given UR"""
        for i, u in enumerate(ur):
            if i == 0:
                pro_ur = u
                pro_len = len(pro_ur)
                pr = Binomial.pmf(pro_len, self._n, self._tht)
                pr *= Uniform.pmf(self.nsegs()) ** pro_len
            else:
                cxt_ur = u
                pr *= self.D.lpmf(pro_ur, cxt_ur)
        return pr

    def calculate_tp(self, ur_old, ur_new):
        """Returns the transition probability of the current and new UR"""
        tp = 1
        for o, n in zip(ur_old, ur_new):
            tp *= self.D.epmf(o, n)
        return tp

    def add_padding(self, ss, token="#"):
        """Pads the string with the specified token"""
        return f"#{ss}#"

    def rm_padding(self, ss, token="#"):
        """Removes the padding from the given string"""
        return re.sub(token, "", ss)

    """ ========== ACCESSORS ============================================ """

    def lxs(self):
        """Returns the lexemes"""
        return self._lxs

    def clxs(self):
        """Returns the lexical sequences of each lexeme"""
        return self._clxs

    def lx2clxs(self, lx: str):
        """Returns a list of lexical contexts given a lexeme"""
        return self._lx2clxs[lx]

    def lx2clx2id(self, lx: str, clx: tuple):
        """Returns the index of a lexical context for a given lexeme"""
        return self._lx2clx2id[lx][clx]

    def lx2nclxs(self, lx: str):
        """Returns the number of lexical contexts given a lexeme"""
        return len(self._lx2clxs[lx])

    def clx2lxs(self, clx: tuple):
        """Returns a list of lexemes given a lexical context"""
        return self._clx2lxs[lx]

    def clx2lx2id(self, clx: tuple, lx: str):
        """Returns the index of a given lexeme for a given lexical context"""
        return self._clx2lx2id[clx][lx]

    def clx2nlxs(self, clx: tuple):
        """Returns the number of lexemes given a lexical context"""
        return len(self._clx2lxs[clx])

    def lx2ur(self, lx: str):
        """Returns the current UR hypothesis for a given lexeme"""
        return self._lx2ur[lx]

    def lx2pr(self, lx: str):
        """Returns the prior probability of the current UR hypothesis"""
        return self._lx2pr[lx]

    def get_hyp(self, lx: str):
        """Returns the current UR hypothesis for a given lexeme"""
        return self.lx2ur(lx), self.lx2pr(lx)

    def get_ur(self, clx: tuple):
        """Returns the current UR for a given lexical context for a lexeme"""
        return np.vstack([self.lx2ur(lx)[self.lx2clx2id(lx, clx)] for lx in clx])

    def get_ur(self, clx: tuple):
        """Returns the current UR for a given lexical context for a lexeme"""
        return "".join([self.lx2ur(lx)[self.lx2clx2id(lx, clx)] for lx in clx])
