import Levenshtein as lv
import numpy as np
import re
import json
from optim.Lexicon import Lexicon
from optim.Phonology import SPE, OT

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                GRAMMAR DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Grammar:
    """========== INITIALIZATION ======================================="""

    def __init__(self, clxs: np.ndarray, srs: np.ndarray, nobs: np.ndarray, lm: float, L, M):
        self.L = L
        self.M = M

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lm = lm

        ## *=*=*= DATA INITIALIZATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._clxs = clxs
        self._srs = srs
        self._nobs = nobs

        ## *=*=*= INDEX DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._clx2id = {clx: i for i, clx in enumerate(clxs)}
        self._sr2id = {sr: i for i, sr in enumerate(srs)}

        ## *=*=*= CACHE DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._hyp2likelihood = {}

    """ ========== INSTANCE METHODS ===================================== """

    def import_likelihoods(self, hyp_filename):
        """Imports a json containing some or all of the likelihoods for a given
        hypothesis
        """
        with open(hyp_filename, "r") as hf:
            self._hyp2likelihood = json.load(hf)

    def export_likelihoods(self, hyp_filename):
        """Exports a json containing some or all of the likelihoods for a given
        hypothesis
        """
        with open(hyp_filename, "w") as hf:
            json.dump(self._hyp2likelihood, hf)

    def predict_srs(self):
        """Generates the SRs for the set of lexical sequences"""
        return [self.predict_sr(clx) for clx in self.clxs()]

    def predict_sr(self, clx):
        """Generates the SR predicted by the Grammar object for a given
        lexical sequence
        """
        ur = self.L.get_ur(clx)
        ur = self.L.add_padding(ur)
        pred_sr = self.M.regex_apply(ur)
        pred_sr = self.L.rm_padding(pred_sr)
        return pred_sr

    def levenshtein(self, pred_sr, obs_sr):
        """Calculates the levenshtein edit distance between the two strings"""
        return np.exp(-lv.distance(pred_sr, obs_sr) * self.lm())

    def compute_likelihoods(self, lx, likelihood):
        """Computes the likelihood of the data for the given lexeme given
        the current set of UR and rule hypotheses
        """
        clxs = self.L.lx2clxs(lx)[1:]
        return np.prod([self.compute_likelihood(clx, likelihood) for clx in clxs])

    def compute_likelihood(self, clx, likelihood):
        """Computes the likelihood of the data for the given lexical context
        given the current set of UR and rule hypotheses
        """
        obs_sr = self.get_sr(clx)
        if obs_sr == "":
            return 1
        pred_sr = self.predict_sr(clx)
        return likelihood(pred_sr, obs_sr) ** self.get_nob(clx)

    def export(self):
        """Exports the current model parameters and predictions"""
        clxs = self.clxs()
        mnames = self.M.get_current_mhyp()
        urs = [self.L.get_ur(clx, s="-") for clx in clxs]
        pred_srs = self.predict_srs()
        obs_srs = self.srs()
        return clxs, mnames, urs, pred_srs, obs_srs

    """ ========== ACCESSORS ============================================ """

    def lm(self):
        """Returns the lambda hyperparameter for the noisy channel"""
        return self._lm

    def clxs(self):
        """Returns the clx of the data"""
        return self._clxs

    def srs(self):
        """Returns the surface forms of the data"""
        return self._srs

    def get_sr(self, clx: tuple):
        """Returns the surface form for the given lexical context"""
        id = self._clx2id[clx]
        return self._srs[id]

    def get_nob(self, clx: tuple):
        """Returns the number of observations for the given form"""
        id = self._clx2id[clx]
        return self._nobs[id]
