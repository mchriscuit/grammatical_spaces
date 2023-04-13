import numpy as np
import copy as cp
from weighted_levenshtein import levenshtein
from models.Lexicon import Lexicon
from models.Phonology import SPE
from models.Inventory import Inventory
from models.Distributions import Uniform, Bernoulli, Geometric, Distance


class Grammar:
    """========== INITIALIZATION ==================================================="""

    def __init__(
        self,
        lxs: np.ndarray,
        cxs: np.ndarray,
        obs: np.ndarray,
        nbs: np.ndarray,
        m_args: tuple,
        i_args: tuple,
        params: tuple,
        pad: str = "#",
    ):
        ## *=*=*= HELPER FUNCTIONS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._vln = np.vectorize(len)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._pad = pad
        self._lm = params["conservativity"]
        self._ml = params["maxlength"]
        self._mb = params["maxbuffer"]
        self._th = params["prolength"]
        self._ps = params["urprior"]
        self._ph = params["urtrans"]
        self._al = params["dfprior"]
        self._bt = params["dftrans"]

        ## *=*=*= BASIC GRAMMAR INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._lxs = lxs
        self._cxs = cxs
        self._obs = obs
        self._nbs = nbs

        ## *=*=*= OBJECT INITIALIZATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        """ (1) Inventory object ==================================================="""
        self._iv = Inventory(*i_args)
        self._ss = self._iv.segs
        self._ns = self._iv.nsegs
        self._fs = self._iv.feats
        self._sm = self._iv.ssim

        """ (2) Distance object ===================================================="""
        self._m = self._ml + self._mb
        self._D = Distance(ss=self._ss, n=self._m)
        self._D.add_derv("sub", self._ml, self._sm)
        self._D.add_pdis("nc", "sub", self._lm, False)
        self._D.add_pdis("pr", "sub", self._ps)
        self._D.add_pdis("tp", "sub", self._ph)

        ## Generate the vectorized weighted levenshtein distance function
        dsc = self._D.sub_dsc
        self._vds = np.vectorize(lambda x, y: levenshtein(x, y, substitute_costs=dsc))

        """ (3) Distribution object ================================================"""
        self._ulen = Geometric(self._ml, self._th)
        self._unif = Uniform(self._ns)
        self._dfpr = Bernoulli(self._al)
        self._dftp = Bernoulli(self._bt)

        """ (4) Lexicon object ====================================================="""
        pdist = (self._ulen, self._unif, self._dfpr, self._dftp, self._D.pr, self._D.tp)
        self.L = Lexicon(self._lxs, self._cxs, self._ml, self._pad, *pdist)

        """ (5) Mappings object ===================================================="""
        self._mnms, self._mdfs = m_args
        self._pfms = (self.pad + self._D.bw.astype(object) + self.pad).astype(str)
        self.M = SPE(self._mnms, self._mdfs, self._pfms, self._iv)

        ## *=*=*= LEXEME GROUPING *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        """ (1) Data indexing ======================================================"""
        self._oid = obs != ""
        self._pid = np.logical_and(self.oid, self.L.bid)

        """ (2) Likelihoods ========================================================"""
        self._exs = self._D.nc.w
        self._exs2lks = self._D.nc.p[:, self._D.nc.w.searchsorted(self.obs)] ** self.nbs

        """ (3) Lexeme to context =================================================="""
        ls = self._lxs.tolist()
        cs = self._cxs.tolist()
        xid = [[i for i, c in enumerate(cs) if l in c] for l in ls]

        """ (4) Lexeme groups ======================================================"""
        gid = []
        gcx = []
        for i, x in enumerate(xid):
            x = cp.copy(x)
            o = True
            for j, y in enumerate(gcx):
                scx, scy = set(x), set(y)
                if len(scx.intersection(scy)) == 0:
                    o = False
                    gid[j] += [i]
                    gcx[j] += x
            if o == True:
                gid.append([i])
                gcx.append(x)
        self._gid = gid
        self._gcx = gcx

    """ ========== CLASS METHODS ================================================== """

    @classmethod
    def rm_padding(cls, ss: np.ndarray, pad="#"):
        """Removes the padding from the given string"""
        return np.char.replace(ss, pad, "")

    """ ========== INSTANCE METHODS =============================================== """

    def rtlikelihood(self, exs: np.ndarray, obs: np.ndarray):
        """Calculates the real-time levenshtein distance between the expected and observed.
        We assume that the contexts are ordered
        """

        ## Calculate the unnormalized likelihoods
        lk = np.exp(-self.lm * self.vds(exs, obs)) ** self.nbs

        return lk

    def chlikelihood(self, exs: np.ndarray, obs: np.ndarray):
        """Calculates the cached levenshtein distance between the expected and observed.
        We assume that the contexts are ordered
        """

        ## Get the indices for the expected forms
        efid = self.exs.searchsorted(exs)

        ## Get the indices for the surface forms
        sfid = np.arange(obs.size)

        ## Get the outputs for each index
        lk = self.exs2lks[efid, sfid]

        return lk

    def likelihood(self, exs: np.ndarray, obs: np.ndarray):
        exs = Grammar.rm_padding(exs).squeeze()
        if self.vln(exs).max() > self.ml:
            return self.rtlikelihood(exs=exs, obs=obs)
        else:
            return self.chlikelihood(exs=exs, obs=obs)

    """ ========== ACCESSORS ===================================================== """

    @property
    def lxs(self):
        return self._lxs

    @property
    def cxs(self):
        return self._cxs

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
    def vln(self):
        return self._vln

    @property
    def vds(self):
        return self._vds

    @property
    def lxs(self):
        return self._lxs

    @property
    def cxs(self):
        return self._cxs

    @property
    def obs(self):
        return self._obs

    @property
    def nbs(self):
        return self._nbs

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
    def exs(self):
        return self._exs

    @property
    def exs2lks(self):
        return self._exs2lks
