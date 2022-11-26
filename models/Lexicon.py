import numpy as np
import re
from copy import deepcopy
from optim.Inventory import Inventory
from optim.Distributions import Binomial, Bernoulli, Uniform, Distance

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                LEXICON DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Lexicon(Inventory):
    """========== INITIALIZATION ======================================="""

    def __init__(
        self,
        lxs: np.ndarray,
        clxs: np.ndarray,
        pars: dict,
        tokens: np.ndarray,
        feats: np.ndarray,
        configs: np.ndarray,
    ):

        self._rng = np.random.default_rng()
        self._pro = ("PROTO",)

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(tokens, feats, configs)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        """Prior"""
        self._ml = pars["max_len"]
        self._th = pars["theta"]
        self._ps = pars["psi"]

        """Proposal"""
        self._me = pars["max_eds"]
        self._ph = pars["phi"]

        """Costs"""
        self._ct = pars["costs"]

        """Initializing distance metric"""
        self.D = Distance(
            self.segs(),
            self._ct,
            self._ml,
            self._me,
            self._ps,
            self._ph
            )

        ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs = lxs
        self._clxs = clxs

        ## *=*=*= LEXICAL DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        """Lexeme to lexical sequence"""
        self._lx2clxs = {x: cxs for x, cxs in zip(self._lxs, self._clxs)}

        """Lexical sequence to lexeme"""
        self._clx2lxs = {
            cx: [x for x in self._lxs if x in cx]
            for cxs in self._clxs for cx in cxs
        }

        ## *=*=*= LEXICON HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._lx2ur = {}
        self._lx2pr = {}

    """ ========== OVERLOADING METHODS =================================== """

    def __deepcopy__(self, memo):
        """Overloads the deepcopy function to only copy the list of
        UR hypotheses and priors, keeping a shallow copy of everything else
        """
        ## Initialize a new class object
        cls = self.__class__
        cp = cls.__new__(cls)
        memo[id(self)] = cp

        ## Create deep copies of the lx2ur and lx2pr attributes
        setattr(cp, "_lx2ur", deepcopy(self.__dict__["_lx2ur"], memo))
        setattr(cp, "_lx2pr", deepcopy(self.__dict__["_lx2pr"], memo))

        return cp

    """ ========== INSTANCE METHODS ===================================== """

    def initialize_urs(self, init_ur_hyp: tuple):
        """Initializes the UR hypotheses for the lexemes"""
        init_lxs, init_clxs, init_urs = init_ur_hyp

        ## Check that number of given UR hypotheses match the number of
        ## lexemes seen in the data
        assert set(self._lxs) == set(
            init_lxs
        ), "Given lexemes and observed lexemes are not equal"

        ## Check that the given lexical contexts match the lexical contexts
        ## seen in the data
        assert all(
            set(self._lx2clxs[lx]) == set(init_clxs[i]) for i, lx in enumerate(init_lxs)
        ), "Given lexical contexts and observed lexical contexts not equal"

        ## Otherwise, update the UR hypothesis and prior
        for init_lx, init_clx, init_ur in zip(init_lxs, init_clxs, init_urs):
            self._lx2ur[init_lx] = {
                clx: "".join(ur) for clx, ur in zip(init_clx, init_ur)
                }
            self._lx2pr[init_lx] = {
                clx: self.compute_pr(self._lx2ur[init_lx][self._pro], ur) if clx != self._pro
                else self.compute_pr(self._lx2ur[init_lx][self._pro])
                for clx, ur in zip(init_clx, init_ur)
                }

    def set_ur(self, lx: np.str_, clx: tuple, ur: np.str_, pr: float):
        """Set the given lexeme in a given lexical sequence to the given UR"""
        self._lx2ur[lx][clx] = ur
        self._lx2pr[lx][clx] = pr

    def set_pro_ur(self, lx: np.str_, ur: np.str_, pr: dict):
        """Set the given lexeme in a given lexical sequence to the given UR"""
        self._lx2ur[lx][self._pro] = ur
        self._lx2pr[lx].update(pr)

    def compute_pr(self, pro_ur, cxt_ur=None):
        """Returns the prior probability of the given UR"""
        pr = 1

        ## If a contextual UR is not specified, calculate the
        ## probability of generating the proto UR
        if cxt_ur is None:
            pro_len = len(pro_ur)
            pr *= Binomial.pmf(pro_len, self._me, self._th)
            pr *= Uniform.pmf(self.nsegs()) ** pro_len
            return pr

        ## Otherwise, calculate the probability of generating the
        ## specified contextual UR given the proto UR
        else:
            pr *= self.D.lpmf(pro_ur, cxt_ur)
            return pr

    def compute_tp(self, o, n):
        """Returns the transition probability between the current and new UR"""
        return self.D.epmf(o, n)

    def sample_pro_ur(self, lx, inplace=True):
        """Samples a prototype underlying form for a given lexeme. Also
        returns the prior of the proposed UR
        """

        ## Retrieve the old prototype UR
        pro_ur_old = self.lx2ur(lx)[self._pro]

        ## Sample a new prototype UR
        pro_ur_new = self.D.erv(pro_ur_old)
        pro_len = len(pro_ur_new)
        pro_pr = Binomial.pmf(pro_len, self._ml, self._th)
        pro_pr *= Uniform.pmf(self.nsegs()) ** pro_len
        pr = {self.pro(): pro_pr}

        ## Calculate the prior for each contextual UR given the
        ## newly-sampled UR
        for clx, cxt_ur in self.lx2ur(lx).items():
            if clx != self.pro():
                cxt_pr = self.compute_pr(pro_ur_new, cxt_ur)
                pr[clx] = cxt_pr

        ## Update the dictionary
        if inplace:
            self.set_pro_ur(lx, pro_ur_new, pr)

        return pro_ur_new, np.prod(list(pr.values()))

    def sample_cxt_ur(self, clx, inplace=True):
        """Samples a UR for a given lexical sequence given the
        proposal distribution. Also returns the prior of the proposed UR
        """
        ur = []
        pr = []

        ## For each lexeme in the lexical item
        for lx in clx:

            ## Retrieve the old underlying form and sample a new underlying form
            ur_old = self.get_lx_clx_ur(lx, clx)
            ur_new = self.D.erv(ur_old)
            ur.append(ur_new)

            ## Calculate the prior of the contextual UR given the prototype UR
            pro_ur = self.get_lx_clx_ur(lx, self._pro)
            pr_new = self.compute_pr(pro_ur, ur_new)
            pr.append(pr_new)

            ## Update the dictionary of URs, if specified
            if inplace:
                self.set_ur(lx, clx, ur_new, pr_new)

        return ur, pr

    def add_padding(self, ss, token="#"):
        """Pads the string with the specified token"""
        return f"#{ss}#"

    def rm_padding(self, ss, token="#"):
        """Removes the padding from the given string"""
        return re.sub(token, "", ss)

    """ ========== ACCESSORS ============================================ """

    ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
    def pro(self):
        """Returns the prototype context"""
        return self._pro

    def lxs(self):
        """Returns the lexemes"""
        return self._lxs

    def clxs(self):
        """Returns the lexical sequences of each lexeme"""
        return self._clxs

    def lx2clxs(self, lx: np.str_):
        """Returns a list of lexical contexts given a lexeme"""
        return self._lx2clxs[lx]

    def lx2nclxs(self, lx: np.str_):
        """Returns the number of lexical contexts given a lexeme"""
        return len(self._lx2clxs[lx])

    def clx2lxs(self, clx: tuple):
        """Returns a list of lexemes given a lexical context"""
        return self._clx2lxs[clx]

    def clx2nlxs(self, clx: tuple):
        """Returns the number of lexemes given a lexical context"""
        return len(self._clx2lxs[clx])

    ## *=*=*= UNDERLYING FORMS (LEXEME LEVEL) *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
    def lx2ur(self, lx: np.str_):
        """Returns the current UR hypothesis for a given lexeme"""
        return self._lx2ur[lx]

    def lx2pr(self, lx: np.str_):
        """Returns the prior probability of the current UR hypothesis"""
        return self._lx2pr[lx]

    def get_hyp(self, lx: np.str_):
        """Returns the UR hypothesis and prior for a given lexical context"""
        return self.lx2ur(lx), self.lx2pr(lx)

    ## *=*=*= UNDERLYING FORMS (LEXICAL CONTEXT LEVEL) *=*=*=*=*=*=*=*=*=*=*
    def get_lx_clx_ur(self, lx: np.str_, clx: tuple):
        """Returns the current UR hypothesis for a given lexeme in a given
        lexical context"""
        return self.lx2ur(lx)[clx]

    def get_lx_clx_pr(self, lx: np.str_, clx: tuple):
        """Returns the prior of the UR hypothesis for a given lexeme in a given
        lexical context"""
        return self.lx2pr(lx)[clx]

    def get_clx_ur(self, clx: tuple):
        """Returns the current UR hypotheses for a given
        lexical context
        """
        return [self.lx2ur(lx)[clx] for lx in clx]

    def get_clx_pr(self, clx: tuple):
        """Returns the prior of the current UR hypotheses for a given
        lexical context
        """
        return [self.lx2pr(lx)[clx] for lx in clx]

    def get_clx_hyp(self, clx: tuple):
        """Returns the UR hypothesis and prior for a given lexical context"""
        return self.get_clx_ur(clx), self.get_clx_pr(clx)
