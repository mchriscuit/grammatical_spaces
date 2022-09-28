import numpy as np
import re
import json
from collections import defaultdict
from models.Inventory import Inventory
from models.Lexicon import Lexicon
from models.Phonology import SPE, OT

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                GRAMMAR DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Grammar:
    """========== INITIALIZATION ======================================="""

    def __init__(self, clxs: list, srs: list, nobs: list, L, M):
        self.L = L
        self.M = M

        ## *=*=*= DATA INITIALIZATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._clxs = clxs
        self._srs = srs
        self._nobs = nobs

        ## *=*=*= INDEX DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._clx2id = {clx: i for i, clx in enumerate(clxs)}
        self._sr2id = {sr: i for i, sr in enumerate(srs)}

        ## *=*=*= CACHE DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._hyp2likelihood = defaultdict(dict)

    """ ========== INSTANCE METHODS ===================================== """

    def import_hyps(self, hyp_filename):
        """Imports a json containing some or all of the likelihoods for a given
        hypothesis
        """
        with open(hyp_filename, "r") as hf:
            self._hyp2likelihood = json.load(hf)

    def export_hyps(self, hyp_filename):
        """Exports a json containing some or all of the likelihoods for a given
        hypothesis
        """
        with open(hyp_filename, "w") as hf:
            json.dump(self._hyp2likelihood, hf)

    def predict_srs(self):
        """Generates the SRs for the set of lexical sequences
        """
        return [self.predict_sr(clx) for clx in self.clxs()]

    def predict_sr(self, clx):
        """Generates the SR predicted by the Grammar object for a given
        lexical sequence
        """
        ur = self.L.get_ur(clx)
        ur_config = self.L.configure(ur)
        pred_sr_config = self.M.apply(ur_config)
        pred_sr = self.L.deconfigure(pred_sr_config)
        return pred_sr

    def compute_likelihood(self, lx):
        """Computes the likelihood of the data for the given lexeme given
        the current set of UR and rule hypotheses
        """
        lhyp_idx = self.L.lhyp_idx(lx)
        mhyp_idx = self.M.mhyp_idx()
        if mhyp_idx in self._hyp2likelihood.get(lhyp_idx, {}):
            return self._hyp2likelihood[lhyp_idx][mhyp_idx]
        likelihood = 1
        clxs = self.L.lx2clxs(lx)
        for clx in clxs:
            pred_sr = self.predict_sr(clx)
            obs_sr = self.clx2sr(clx)
            likelihood *= int(obs_sr == pred_sr) ** self.clx2nob(clx)
        self._hyp2likelihood[lhyp_idx][mhyp_idx] = likelihood
        return self._hyp2likelihood[lhyp_idx][mhyp_idx]

    def export(self):
        """Exports the current model parameters and predictions"""
        clxs = self.clxs()
        mnames = self.M.get_current_mhyp()
        urs = [self.L.get_ur(clx) for clx in clxs]
        pred = self.predict_srs()
        srs = self.srs()
        return clxs, mnames, urs, pred, srs

    """ ========== ACCESSORS ============================================ """

    def clxs(self):
        """Returns the lexical sequences of the data"""
        return self._clxs

    def srs(self):
        """Returns the surface forms of the data"""
        return self._srs

    def nobs(self):
        """Returns the number of observations for each surface
        form in the data
        """
        return self._nobs

    def clx2nob(self, clx: tuple):
        """Returns the number of times we see a given lexical sequence"""
        return self._nobs[self._clx2id[clx]]

    def clx2sr(self, clx: tuple):
        """Returns the surface form of a given lexical sequence"""
        return self._srs[self._clx2id[clx]]
