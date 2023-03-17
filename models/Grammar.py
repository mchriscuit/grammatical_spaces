import numpy as np
import re
import copy as cp
import itertools as it

from Levenshtein import distance
from optim.Lexicon import Lexicon
from optim.Phonology import SPE


class Grammar:
    """========== INITIALIZATION ================================================"""

    def __init__(
        self,
        lm: float,
        lxs: np.ndarray,
        clxs: np.ndarray,
        srs: np.ndarray,
        ns: np.ndarray,
        lkwargs: dict,
        mkwargs: dict,
        ikwargs: dict,
    ):
        ## *=*=*= CLASS INITIALIZATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self.M = SPE(**mkwargs)

        assert False

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lm = lm

        ## *=*=*= DATA INITIALIZATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs = lxs
        self._clxs = clxs
        self._srs = srs
        self._nbs = ns

        ## *=*=*= INDEX DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._cxt2id = {clx: i for i, clx in enumerate(clxs)}
        self._sr2id = {sr: i for i, sr in enumerate(srs)}

        assert False

    """ ========== OVERLOADING METHODS ================================== """

    def __deepcopy__(self, memo: dict):
        """Overloads the deepcopy function to only copy the Mapping and
        Lexicon objects, keeping a shallow copy of everything else
        """

        ## Initialize a new class object
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__dict__.update(self.__dict__)
        memo[id(self)] = cp

        ## Create deep copies of the L and M attributes
        # setattr(cp, "L", deepcopy(self.__dict__["L"], memo))
        # setattr(cp, "M", deepcopy(self.__dict__["M"], memo))

        return cp

    """ ========== INSTANCE METHODS ===================================== """

    ## *=*=*= DISTANCE METRIC *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

    def levenshtein(self, pred_sr: str, obs_sr: str):
        """Calculates the levenshtein edit distance between the two strings"""
        return distance(pred_sr, obs_sr)

    def exponentiate(self, distances: np.ndarray):
        """Computes the negative exponentiated noisy channel value"""
        return np.exp(-distances * self.lm())

    ## *=*=*= GENERATING PREDICTIONS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    def predict_sr(self, str_ur, mhyp_idx=None):
        """Generates the SR predicted by the Grammar object for a given
        lexical sequence. If a mapping hypothesis is given, us it. Otherwise,
        use the currently-set mapping hypothesis
        """
        return self.L.rm_padding(
            self.M.regex_apply(self.L.add_padding(str_ur), mhyp_idx)
        )

    ## *=*=*= LIKELIHOOD CALCULATION =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    def compute_cxt_likelihoods(
        self, str_urs: list, cxts: list, fdistance, mhyp_idxs=[]
    ):
        """Computes the likelihood of the given contexts for
        the given lexeme given the set of UR and rule hypotheses
        """

        ## If no mapping hypothesis is specified, retrieve
        ## the current hypothesis
        mhyp_idxs = (
            [self.M.get_current_mhyp_idx()] if len(mhyp_idxs) == 0 else mhyp_idxs
        )

        ## Retrieve the number of observations for each form
        nobs = np.array([self.get_nob(cxt) for cxt in cxts])

        ## Calculate the likelihood
        likelihood = np.array(
            [
                [
                    fdistance(self.predict_sr(str_ur, mhyp_idx), self.get_sr(cxt))
                    if self.get_sr(cxt) != ""
                    else 1.0
                    for str_ur, cxt in zip(str_urs, cxts)
                ]
                for mhyp_idx in mhyp_idxs
            ]
        )
        return np.prod(
            self.exponentiate(likelihood) ** np.array(nobs), axis=1
        ).squeeze()

    def compute_all_likelihoods(self, fdistance, mhyp_idxs=[]):
        """Computes the likelihood of the data given the current UR and
        rule hypotheses
        """

        ## Retrieve the URs and contexts of the observed data
        cxts = self.cxts()
        str_cxt_urs = [self.L.str_cxt_ur(cxt) for cxt in cxts]

        ## Compute the likelihood for all the URs
        return self.compute_cxt_likelihoods(str_cxt_urs, cxts, fdistance, mhyp_idxs)

    ## *=*=*= EXPORTING GRAMMAR =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    def export(self):
        """Exports the current model parameters and predictions"""

        ## Export the clxs
        cxts = ["-".join(cxt) for cxt in self.cxts()]

        ## Export the rule names
        mnames = self.M.get_current_mhyp()

        ## Get the URs
        urs = [self.L.str_cxt_ur(cxt, "-") for cxt in self.cxts()]

        ## Get the predicted SRs
        pred_srs = [self.predict_sr(re.sub("-", "", ur)) for ur in urs]

        ## Get the observed SRs
        obs_srs = self.srs()

        return cxts, mnames, urs, pred_srs, obs_srs

    """ ========== ACCESSORS ============================================ """

    def lm(self):
        """Returns the lambda hyperparameter for the noisy channel"""
        return self._lm

    def lxs(self):
        """Returns the lexemes of the data"""
        return self.L.lxs()

    def cxts(self):
        """Returns the clx of the data"""
        return self._cxts

    def srs(self):
        """Returns the surface forms of the data"""
        return self._srs

    def get_sr(self, cxt: tuple):
        """Returns the surface form for the given lexical context"""
        id = self._cxt2id[cxt]
        return self._srs[id]

    def get_nob(self, cxt: tuple):
        """Returns the number of observations for the given form"""
        id = self._cxt2id[cxt]
        return self._nobs[id]
