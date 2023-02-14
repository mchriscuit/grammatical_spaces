import numpy as np
import re
from copy import copy, deepcopy
from optim.Inventory import Inventory
from optim.Distributions import Binomial, Geometric, Bernoulli, Uniform, Distance

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
        """Prior Distribution"""
        self._ml = pars["max_len"]
        self._th = pars["theta"]
        self._ps = pars["psi"]
        self._al = pars["alpha"]

        """Initializing Geometric Distribution"""
        self.G = Geometric(self._ml, self._th)

        """Proposal Distribution"""
        self._ph = pars["phi"]
        self._bt = pars["beta"]

        """Initializing Distance Metric"""
        self.D = Distance(self.segs(), self._ml, self._ps, self._ph)

        ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs = lxs
        self._clxs = clxs
        self._lx2clxs = {x: cxs for x, cxs in zip(self._lxs, self._clxs)}

        ## *=*=*= LEXICON HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._lx2ur = {}
        self._lx2pr = {}

    """ ========== OVERLOADING METHODS =================================== """

    def __copy__(self):
        """Overloads the copy function"""
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__dict__.update(self.__dict__)
        return cp

    def __deepcopy__(self, memo):
        """Overloads the deepcopy function to only copy the list of
        UR hypotheses and priors, keeping a shallow copy of everything else
        """

        ## Initialize a new class object
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__dict__.update(self.__dict__)
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
            ## First, update the dictionary for all the URs for all contexts
            ## Go through each of the contextual URs and denote whether
            ## the UR matches the proto-UR: mark it as such if so
            self._lx2ur[init_lx] = {
                init_cx: init_u[0]
                if self.is_pro(init_cx)
                else (init_u[0], int(init_u[1]))
                for init_cx, init_u in zip(init_clx, init_ur)
            }

            ## Calculate the prior of each UR as well
            self._lx2pr[init_lx] = {
                init_cx: self.compute_pr(self.pro_ur(init_lx))
                if self.is_pro(init_cx)
                else self.compute_pr(self.pro_ur(init_lx), init_u[0], init_u[1])
                for init_cx, init_u in self._lx2ur[init_lx].items()
            }

    """ ========== SAMPLING METHODS ===================================== """

    def sample_default(self, default_old):
        """Samples whether a contextual UR is a default UR"""
        return abs(default_old - Bernoulli.rv(self._bt))

    def sample_pro_ur(self, lx, inplace=True):
        """Samples a prototype underlying form for a given lexeme. Also
        returns the prior of the proposed UR
        """

        ## Retrieve the old prototype UR
        pro_ur_old = self.pro_ur(lx)

        ## Sample a new prototype UR
        pro_ur_new = self.D.tprv(pro_ur_old)

        ## Calculate the prior for the new prototype UR
        pro_pr = self.compute_pr(pro_ur_new)

        ## Update the dictionary
        ur = {self.pro(): pro_ur_new}
        pr = {self.pro(): pro_pr}

        ## Loop through each contextual UR
        for clx in self.lx2clxs(lx):
            ## Make sure that it is a contextual UR
            if self.is_cxt(clx):
                clx_ur, clx_default = self.lx_clx_ur(lx, clx)

                ## If it is marked as default, change the UR to match
                ## the proto UR
                if clx_default:
                    clx_ur = pro_ur_new

                ## Calculate the prior for each contextual UR given the newly-sampled UR
                clx_pr = self.compute_pr(pro_ur_new, clx_ur, clx_default)

                ## Update the dictionary
                ur[clx] = (clx_ur, clx_default)
                pr[clx] = clx_pr

        ## Check if inplace: if so, update the cache
        if inplace:
            self.set_ur(lx, ur, pr)

        return ur, pr

    def sample_cxt_ur(self, clx, inplace=True):
        """Samples a UR for a given lexical sequence given the
        proposal distribution. Also returns the prior of the proposed UR
        """
        cxt_ur_new = []
        cxt_pr_new = []

        ## For each lexeme in the lexical item
        for lx in clx:
            ## Retrieve the prototype UR
            pro_ur = self.pro_ur(lx)

            ## Retrieve the old contextual UR
            lx_clx_ur_old, lx_clx_default_old = self.lx_clx_ur(lx, clx)

            ## Sample a new default value
            lx_clx_default_new = self.sample_default(lx_clx_default_old)

            ## Sample a new contextual UR given the default value
            if lx_clx_default_new:
                lx_clx_ur_new = pro_ur
            else:
                lx_clx_ur_new = self.D.tprv(lx_clx_ur_old)

            ## Calculate the prior given the sampled default value and
            ## prototype underlying form
            lx_clx_pr_new = self.compute_pr(pro_ur, lx_clx_ur_new, lx_clx_default_new)

            ## Append the sample values
            lx_clx_ur_new = (lx_clx_ur_new, lx_clx_default_new)
            cxt_ur_new.append(lx_clx_ur_new)
            cxt_pr_new.append(lx_clx_pr_new)

            ## Update the dictionary of URs, if specified
            if inplace:
                self.set_clx_ur(lx, clx, lx_clx_ur_new, lx_clx_pr_new)

        return cxt_ur_new, cxt_pr_new

    def compute_pr(self, pro_ur, cxt_ur=None, cxt_id=0):
        """Returns the prior probability of the given UR"""
        pr = 1

        ## If a contextual UR is not specified, calculate the
        ## probability of generating the proto UR
        if cxt_ur is None:
            pro_len = len(pro_ur)
            pr *= self.G.pmf(pro_len)
            pr *= Uniform.pmf(self.nsegs()) ** pro_len
            return pr

        ## Otherwise, calculate the probability of generating the
        ## specified contextual UR given the proto UR
        else:
            ## If cxt_id = 0, then generate a contextual underlying form
            pr *= (1 - Bernoulli.pmf(1, self._al)) ** (1 - cxt_id)
            pr *= self.D.prpmf(pro_ur, cxt_ur) ** (1 - cxt_id)

            ## If cxt_id = 1, then check that the prototype underlying form
            ## an the contextual underlying form are identical
            pr *= Bernoulli.pmf(1, self._al) ** cxt_id
            pr *= float(pro_ur == cxt_ur) ** cxt_id
            return pr

    def compute_pro_tp(self, ur_old, ur_new):
        """Returns the transition probability between the current and new UR"""
        return self.D.tppmf(ur_old, ur_new)

    def compute_cxt_tp(self, ur_old, ur_new, id_old, id_new):
        """Returns the transition probability between the current and new UR"""
        if id_old != id_new:
            tp = Bernoulli.pmf(1, self._bt)
        else:
            tp = 1 - Bernoulli.pmf(1, self._bt)
        tp *= self.D.tppmf(ur_old, ur_new)
        return tp

    """ ========== MUTATORS ============================================= """

    def set_ur(self, lx: np.str_, ur: dict, pr: dict):
        """Set the given lexeme to the given URs"""
        self._lx2ur[lx].update(ur)
        self._lx2pr[lx].update(pr)

    def set_pro_ur(self, lx: np.str_, ur: np.str_, pr: dict):
        """Set the given lexeme to the given prototype UR"""
        self._lx2ur[lx][self.pro()] = ur
        self._lx2pr[lx].update(pr)

    def set_clx_ur(self, lx: np.str_, clx: tuple, ur: np.str_, pr: float):
        """Set the given lexeme in a given lexical sequence to the given UR"""

        ## Check whether the provided clx is not PROTO
        assert self.is_cxt(clx), f"Provided context is not valid: {clx}"

        ## Check whether the prototype UR is identical to the UR to be set
        self._lx2ur[lx][clx] = ur

        ## Update the probability of that contextual UR
        self._lx2pr[lx][clx] = pr

    def add_padding(self, ss, token="#"):
        """Pads the string with the specified token"""
        return f"#{ss}#"

    def rm_padding(self, ss, token="#"):
        """Removes the padding from the given string"""
        return re.sub(token, "", ss)

    """ ========== ACCESSORS ============================================ """

    ## *=*=*= BOOLEAN OPERATORS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*+*=*=*
    def is_pro(self, clx: tuple):
        """Checks whether the clx is the prototype context"""
        return clx == self.pro()

    def is_cxt(self, clx: tuple):
        """Checks whether the clx is the contextual context"""
        return not self.is_pro(clx)

    def is_default(self, lx: np.str_, ur: np.str_):
        """Checks whether the ur is equal to the prototype UR"""
        return self.pro_ur(lx) == ur

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

    ## *=*=*= UNDERLYING FORM ACCESSORS  *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
    def lx_ur(self, lx: np.str_):
        """Returns the URs for a given lexeme"""
        return self._lx2ur[lx]

    def lx_pr(self, lx: np.str_):
        """Returns a dictionary containing the priors for all clxs of a given lexeme"""
        return self._lx2pr[lx]

    def lx_clx_ur(self, lx: np.str_, clx: tuple):
        """Returns the UR for a lexeme in a given clx"""
        return self._lx2ur[lx][clx]

    def lx_clx_pr(self, lx: np.str_, clx: tuple):
        """Returns the prior for the UR for a lexeme in a given clx"""
        return self._lx2pr[lx][clx]

    ## (1) Prototype URs
    def pro_ur(self, lx: np.str_):
        """Returns the prototype UR for the given lexeme"""
        return self.lx_clx_ur(lx, self.pro())

    def pro_pr(self, lx: np.str_):
        """Returns the probability of the prototype UR for the given lexeme"""
        return self.lx_clx_pr(lx, self.pro())

    ## (2) Contextual URs
    def cxt_ur_full(self, clx: tuple):
        """Returns a list containing the URs of all the lexemes in a clx"""
        return [self.lx_clx_ur(lx, clx) for lx in clx]

    def cxt_ur(self, clx: tuple):
        """Returns a list containing the URs of all the lexemes in a clx"""
        return [self.lx_clx_ur(lx, clx)[0] for lx in clx]

    def str_cxt_ur(self, clx: tuple, sep=""):
        """Returns the concatenated URs of all the lexemes in a clx"""
        return sep.join(self.cxt_ur(clx))

    def cxt_pr(self, clx: tuple):
        """Returns a list containing the priors of the URs for all the lexemes in a clx"""
        return [self.lx_clx_pr(lx, clx) for lx in clx]

    def prd_pr(self, clx: tuple):
        """Returns the product of the priors of the URs for all the lexemes in a clx"""
        return np.prod(self.cxt_pr(clx))
