import numpy as np
import copy as cp

from polyleven import levenshtein
from utils.loader import Loader
from models.Lexicon import Lexicon
from models.Phonology import SPE


class Grammar:
    """========== INITIALIZATION ================================================"""

    def __init__(
        self,
        ikw: dict,
        lkw: dict,
        mkw: dict,
        lxs: np.ndarray,
        cxs: np.ndarray,
        srs: np.ndarray,
        nbs: np.ndarray,
        pad: str = "#",
    ):

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lm = lkw["lambda"]
        self._pad = pad

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._vln = np.vectorize(len)
        self._vds = np.vectorize(levenshtein)

        """Lexicon ================================"""
        with Loader("(1) Initializing the ``Lexicon`` object..."):
            self.L = Lexicon(ikw, lkw, lxs, cxs)

        """Mappings ==============================="""
        with Loader("(2) Initializing the ``SPE`` object..."):
            self.M = SPE(ikw, **mkw)
            forms = f"{self.pad}" + self.L.dist.R.astype(object) + f"{self.pad}"
            forms = forms.astype(str)
            self.M.preprocess_tf(forms)

        ## *=*=*= BASIC GRAMMAR INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs = lxs
        self._cxs = cxs
        self._srs = srs
        self._nbs = nbs

        """SR indexing ============================"""
        self._oid = srs != ""
        self._pid = np.logical_and(self.oid, self.L.bid)

        """Likelihoods ============================"""
        self._ex, lk = self.L.dist.O, self.L.dist.ncpmf
        self._ml = self.vln(self.ex).max()
        self._ex2lk = lk[:, self.ex.searchsorted(self.srs)] ** self.nbs

        ## *=*=*= LEXEME GROUPING *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._gid = []
        self._gcx = []

        """Lexeme to context ======================"""
        l2c = [[i for i, cx in enumerate(self.cxs) if lx in cx] for lx in self.lxs]

        """Groups of lexemes ======================"""
        for i, c in enumerate(l2c):
            c = cp.copy(c)
            overlap = True
            for j, x in enumerate(self._gcx):
                if len(set(c).intersection(set(x))) == 0:
                    self._gid[j] += [i]
                    self._gcx[j] += c
                    overlap = False
            if overlap:
                self._gid.append([i])
                self._gcx.append(c)

    """ ========== CLASS METHODS =============================================== """

    @classmethod
    def rm_padding(cls, ss: np.ndarray, pad="#"):
        """Removes the padding from the given string"""
        return np.char.replace(ss, pad, "")

    """ ========== ACCESSORS ================================================== """

    @property
    def ml(self):
        return self._ml

    @property
    def lm(self):
        return self._lm

    @property
    def pad(self):
        return self._pad

    @property
    def lxs(self):
        return self._lxs

    @property
    def cxs(self):
        return self._cxs

    @property
    def srs(self):
        return self._srs

    @property
    def oid(self):
        return self._oid

    @property
    def pid(self):
        return self._pid

    @property
    def gid(self):
        return self._gid

    @property
    def nbs(self):
        return self._nbs

    @property
    def ex(self):
        return self._ex

    @property
    def ex2lk(self):
        return self._ex2lk

    @property
    def vln(self):
        return self._vln

    @property
    def vds(self):
        return self._vds

    """ ========== INSTANCE METHODS ============================================ """

    def rtlikelihood(self, exs: np.ndarray, srs: np.ndarray):
        """Calculates the real-time levenshtein distance between the expected and observed.
        We assume that the contexts are ordered
        """

        ## Calculate the unnormalized likelihoods
        lk = np.exp(-self.lm * self.vds(exs, srs)) ** self.nbs

        return lk

    def chlikelihood(self, exs: np.ndarray, srs: np.ndarray):
        """Calculates the cached levenshtein distance between the expected and observed.
        We assume that the contexts are ordered
        """

        ## Get the indices for the expected forms
        efid = self.ex.searchsorted(exs)

        ## Get the indices for the surface forms
        sfid = np.arange(len(srs))

        ## Get the outputs for each index
        lk = self.ex2lk[efid, sfid]

        return lk

    def likelihood(self, exs: np.ndarray, srs: np.ndarray):
        exs = Grammar.rm_padding(exs).squeeze()
        if self.vln(exs).max() > self.ml:
            return self.rtlikelihood(exs=exs, srs=srs)
        else:
            return self.chlikelihood(exs=exs, srs=srs)
