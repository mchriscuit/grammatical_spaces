import numpy as np
import re
import copy as cp
import itertools as it

from optim.Inventory import Inventory
from optim.Distributions import Geometric, Bernoulli, Uniform, Distance


class Lexicon(Inventory):
    """========== INITIALIZATION ================================================"""

    def __init__(self, lxs: np.ndarray, clxs: np.ndarray, ikwargs: dict, hkwargs: dict):
        self._rng = np.random.default_rng()
        self._pro = ("PROTO",)

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(**ikwargs)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        """Prior Distribution"""
        self._ml = hkwargs["max_len"]
        self._th = hkwargs["theta"]
        self._ps = hkwargs["psi"]
        self._al = hkwargs["alpha"]

        """Proposal Distribution"""
        self._ph = hkwargs["phi"]
        self._bt = hkwargs["beta"]

        """Initializing Geometric Distribution"""
        self.G = Geometric(self._ml, self._th)

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
        # setattr(cp, "_lx2ur", deepcopy(self.__dict__["_lx2ur"], memo))
        # setattr(cp, "_lx2pr", deepcopy(self.__dict__["_lx2pr"], memo))

        return cp

    """ ========== CLASS METHODS ======================================= """

    @classmethod
    def add_padding(cls, ss, pad="#"):
        """Pads the string with the specified token"""
        return f"{pad}{ss}{pad}"

    @classmethod
    def rm_padding(cls, ss, pad="#"):
        """Removes the padding from the given string"""
        return ss.replace(pad, "")

    """ ========== INSTANCE METHODS ===================================== """

    ## *=*=*= INITIALIZATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
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

            ## Calculate the prior of each prototype and contextual UR as well
            self._lx2pr[init_lx] = {
                init_cx: self.compute_pr(self.pro_ur(init_lx))
                if self.is_pro(init_cx)
                else self.compute_pr(self.pro_ur(init_lx), init_u[0], init_u[1])
                for init_cx, init_u in self._lx2ur[init_lx].items()
            }

    ## *=*=*= SAMPLING METHODS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

    def sample_default(self, default: int):
        """Samples whether a contextual UR is a default UR"""
        return abs(default - Bernoulli.rv(self._bt))

    def sample_pro_ur(self, lx: np.str_, cxts: list, inplace: bool = False):
        """Samples a prototype underlying form for a given lexeme. Also
        returns the prior of the proposed UR
        """

        ## Initialize empty dictionaries for the prototype underlying form
        ur_new = {}
        pr_new = {}

        ## Initialize the concatenated UR and
        ## product prior probability
        str_urs = []
        prd_prs = 1

        ## Retrieve the old prototype UR
        pro_ur_old = self.pro_ur(lx)

        ## Sample a new prototype UR
        pro_ur_new = self.D.tprv(pro_ur_old)

        ## Calculate the prior for the new prototype UR
        pro_pr = self.compute_pr(pro_ur_new)
        prd_prs *= pro_pr

        ## Update the dictionary
        ur_new[lx] = {self.pro(): pro_ur_new}
        pr_new[lx] = {self.pro(): pro_pr}

        ## Loop through each contextual UR
        for cxt in cxts:

            ## Get the current contextual UR and default setting
            cxt_ur, cxt_default = self.lx_clx_ur(lx, cxt)

            ## If it is marked as default, change the UR to
            ## match the prototype UR
            if cxt_default:
                cxt_ur = pro_ur_new

            ## Calculate the prior for each contextual UR given the newly-sampled UR
            cxt_pr = self.compute_pr(pro_ur_new, cxt_ur, cxt_default)

            ## Update the dictionary
            ur_new[lx][cxt] = (cxt_ur, cxt_default)
            pr_new[lx][cxt] = cxt_pr

            ## Update the URs and priors
            str_urs.append(
                "".join([self.lx_clx_ur(x, cxt)[0] if x != lx else cxt_ur for x in cxt])
            )
            prd_prs *= cxt_pr

        ## Update the dictionary of URs, if specified
        if inplace:
            self.set_ur(ur_new, pr_new)

        return pro_ur_new, ur_new, pr_new, str_urs, prd_prs

    def sample_cxt_ur(self, clx: tuple, inplace: bool = False):
        """Samples a UR for a given lexical sequence given the
        proposal distribution. Also returns the prior of the proposed UR
        """

        ## Initialize empty dictionaries for the contextual underlying form
        cxt_ur_new = {}
        cxt_pr_new = {}

        ## Initialize the concatenated UR and
        ## product prior probability
        str_ur = ""
        prd_pr = 1

        ## For each lexeme in the lexical item
        for lx in clx:

            ## Initialize the inner dictionaries
            cxt_ur_new[lx] = {}
            cxt_pr_new[lx] = {}

            ## Retrieve the prototype UR
            pro_ur = self.pro_ur(lx)

            ## Retrieve the old contextual UR for that lexeme
            lx_clx_ur_old, lx_clx_default_old = self.lx_clx_ur(lx, clx)

            ## Sample a new default value for the contextual UR for that lexeme
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
            cxt_ur_new[lx][clx] = (lx_clx_ur_new, lx_clx_default_new)
            cxt_pr_new[lx][clx] = lx_clx_pr_new

            ## Update the ur and prior
            str_ur += lx_clx_ur_new
            prd_pr *= lx_clx_pr_new

        ## Update the dictionary of URs, if specified
        if inplace:
            self.set_ur(cxt_ur_new, cxt_pr_new)

        return cxt_ur_new, str_ur, cxt_pr_new, prd_pr

    def compute_pr(self, pro_ur: np.str_, cxt_ur: np.str_ = None, cxt_id: int = 0):
        """Returns the prior probability of the given UR"""

        ## Initialize the prior
        pr = 1

        ## If a contextual UR is not specified, calculate the
        ## probability of generating the prototype UR
        if cxt_ur is None:

            ## Retrieve the length of the prototype UR
            ## and calculate the prior of generating a prototype
            ## UR of that length
            pro_len = len(pro_ur)
            pr *= self.G.pmf(pro_len)
            pr *= Uniform.pmf(self.nsegs()) ** pro_len

        ## Otherwise, calculate the probability of generating the
        ## specified contextual UR given the prototype UR
        else:

            ## If cxt_id = 0, then generate a contextual underlying form
            pr *= (1 - Bernoulli.pmf(1, self._al)) ** (1 - cxt_id)
            pr *= self.D.prpmf(pro_ur, cxt_ur) ** (1 - cxt_id)

            ## If cxt_id = 1, then check that the prototype underlying form
            ## an the contextual underlying form are identical
            pr *= Bernoulli.pmf(1, self._al) ** cxt_id
            pr *= float(pro_ur == cxt_ur) ** cxt_id

        return pr

    def compute_pro_tp(self, ur_old: np.str_, ur_new: np.str_):
        """Returns the transition probability between the current
        and new prototype UR for a given lexeme
        """
        return self.D.tppmf(ur_old, ur_new), self.D.tppmf(ur_new, ur_old)

    def compute_lx_cxt_tp(self, lx_cxt_old: tuple, lx_cxt_new: tuple):
        """Returns the transition probability between the current
        and new contextual UR for a given lexeme"""

        ## Retrieve the URs and identity settings for each lexeme
        ur_old, id_old = lx_cxt_old
        ur_new, id_new = lx_cxt_new

        ## Initialize transition probability
        tp = 1

        ## Calculate the probability of switching identity
        ## parameter settings
        id = int(id_old != id_new)
        tp *= Bernoulli.pmf(1, self._bt) ** id
        tp *= (1 - Bernoulli.pmf(1, self._bt)) ** (1 - id)

        ## Regardless of parameter setting, calculate the probability of
        ## transitioning from the old contextual UR to the new one
        tp *= self.D.tppmf(ur_old, ur_new)

        return tp

    def compute_clx_cxt_tp(self, clx_cxt_old: dict, clx_cxt_new: dict):
        """Returns the transition probability between the current
        and new contextual UR"""

        ## Initialize the transition probabilities
        tp_old_new = 1
        tp_new_old = 1

        ## Loop through each lexeme in the clx and calculate the probabilities
        for lx, clx_ur in clx_cxt_old.items():
            for clx in clx_ur:

                ## Retrieve the relevant contextual UR and parameter setting
                clx_ur_old = clx_cxt_old[lx][clx]
                clx_ur_new = clx_cxt_new[lx][clx]

                ## Calculate the probability of transition from the old
                ## contextual UR to the new contextual UR and vice versa
                tp_old_new *= self.compute_lx_cxt_tp(clx_ur_old, clx_ur_new)
                tp_new_old *= self.compute_lx_cxt_tp(clx_ur_new, clx_ur_old)

        return tp_old_new, tp_new_old

    ## *=*=*= MUTATORS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

    def set_ur(self, ur: dict = {}, pr: dict = {}):
        """Set the (subset of) URs and priors to the given dictionaries"""
        for lx in ur:
            self._lx2ur[lx].update(ur[lx])
            self._lx2pr[lx].update(pr[lx])

    ## *=*=*= ACCESSORS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    ## ====== BOOLEAN OPERATORS =========================================

    def is_pro(self, clx: tuple):
        """Checks whether the clx is the prototype context"""
        return clx == self.pro()

    def is_cxt(self, clx: tuple):
        """Checks whether the clx is the contextual context"""
        return not self.is_pro(clx)

    def is_id(self, lx: np.str_, ur: np.str_):
        """Checks whether the UR is equal to the prototype UR"""
        return self.pro_ur(lx) == ur

    ## ====== LEXICAL INFORMATION =======================================

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

    def lx2cxts(self, lx: np.str_):
        """Returns a list of lexical contexts given a lexeme"""
        return [cx for cx in self._lx2clxs[lx] if self.is_cxt(cx)]

    ## ====== UNDERLYING FORMS ==========================================

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

    def pro_ur(self, lx: np.str_):
        """Returns the prototype UR for the given lexeme"""
        return self.lx_clx_ur(lx, self.pro())

    def pro_pr(self, lx: np.str_):
        """Returns the probability of the prototype UR for the given lexeme"""
        return self.lx_clx_pr(lx, self.pro())

    def cxt_ur(self, clx: tuple):
        """Returns a dictionary containing the URs of all the lexemes in a clx"""
        return {lx: {clx: self.lx_clx_ur(lx, clx)} for lx in clx}

    def cxt_pr(self, clx: tuple):
        """Returns a dictionary containing the priors of the URs for all the lexemes in a clx"""
        return {lx: {clx: self.lx_clx_pr(lx, clx)} for lx in clx}

    def get_cxt_info(self, clx: tuple):
        """Returns the contextual UR and prior for a given clx"""
        return (
            self.cxt_ur(clx),
            self.str_cxt_ur(clx),
            self.cxt_pr(clx),
            self.prd_cxt_pr(clx),
        )

    def str_cxt_ur(self, cxt: tuple, sep: np.str_ = ""):
        """Returns the concatenated URs of all the lexemes in a cxt"""
        return sep.join([self.lx_clx_ur(lx, cxt)[0] for lx in cxt])

    def prd_cxt_pr(self, cxt: tuple):
        """Returns the product of the priors of the URs for all the lexemes in a cxt"""
        return np.prod([self.lx_clx_pr(lx, cxt) for lx in cxt])
