import numpy as np

from models.Inventory import Inventory
from models.Distributions import Geometric, Bernoulli, Uniform, Distance


class Lexicon(Inventory):
    """========== INITIALIZATION ================================================"""

    def __init__(
        self, ikw: dict, lkw: dict, lxs: np.ndarray, cxs: np.ndarray, pad: str = "#"
    ):

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(**ikw)
        self._rng = np.random.default_rng()
        self._vln = np.vectorize(len)

        ## *=*=*= DISTRIBUTIONS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

        """Prior Distribution ====================="""
        self._ml = lkw["maxLen"]
        self._lm = lkw["lambda"]
        self._th = lkw["theta"]
        self._ps = lkw["psi"]
        self._al = lkw["alpha"]

        """Proposal Distribution =================="""
        self._ph = lkw["phi"]
        self._bt = lkw["beta"]

        """Geometric Distribution ================="""
        self._geom = Geometric(self._ml, self._th)

        """Distance Distribution =================="""
        self._dist = Distance(self.segs, self._ml, self._lm, self._ps, self._ph)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._snm = -1
        self._enc = "U" + f"{self.ml}"
        self._pad = pad

        ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs, self._nlxs = lxs, len(lxs)
        self._cxs, self._ncxs = cxs, len(cxs)

        ## *=*=*= LEXICON HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._bid = np.array([[lx in cx for cx in self.cxs] for lx in self.lxs])
        self._cid = np.nonzero(self.bid)

        self._pdim = (self.nlxs,)
        self._cdim = (self.nlxs, self.ncxs)

        """Pro underlying form (nlx, ) ============"""
        self._pro = np.full(self._pdim, "", dtype=self.enc)
        self._pro = self._pro  ## TODO: Initialize protytpe undelying forms

        """Cxt underlying form (nlx, ncx) ========="""
        self._cxt = np.full(self._cdim, "", dtype=self.enc)
        self._cxt = self._cxt  ## TODO: Initialize contextual underlying forms

        """Identity parameters (nlx, ncx) ========="""
        self._idy = np.full(self._cdim, self.snm, dtype=int)
        self._idy[self.cid] = 1  ## TODO: Initialize identity parameters

        """Underlying form priors ================="""
        self._prpro = self.pr(self.pro)
        self._prcxt = self.pr(self.pro, self.cxt, self.idy)

    """ ========== ACCESSORS ================================================== """

    @property
    def geom(self):
        return self._geom

    @property
    def dist(self):
        return self._dist

    @property
    def ml(self):
        return self._ml

    @property
    def snm(self):
        return self._snm

    @property
    def enc(self):
        return self._enc

    @property
    def pad(self):
        return self._pad

    @property
    def lxs(self):
        return self._lxs

    @property
    def nlxs(self):
        return self._nlxs

    @property
    def cxs(self):
        return self._cxs

    @property
    def ncxs(self):
        return self._ncxs

    @property
    def pro(self):
        return self._pro

    @property
    def cxt(self):
        return self._cxt

    @property
    def prpro(self):
        return self._prpro

    @property
    def prcxt(self):
        return self._prcxt

    @property
    def idy(self):
        return self._idy

    @property
    def bid(self):
        return self._bid

    @property
    def cid(self):
        return self._cid

    @property
    def al(self):
        return self._al

    @property
    def bt(self):
        return self._bt

    @property
    def vln(self):
        return self._vln

    def join(self, cxt: np.ndarray, pad: str = None):
        pad = self.pad if pad is None else pad
        jxt = f"{pad}" + cxt.astype(object).sum(axis=0) + f"{pad}"
        return jxt.astype(str)

    """ ========== INSTANCE METHODS ============================================ """

    ## *=*=*= PROBABILITY METHODS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

    def pr(self, pro: np.ndarray, cxt: np.ndarray = None, idy: np.ndarray = None):

        ## If no contextual underlying form vector is given, calculate the
        ## prior of the prototype underlying form
        if cxt is None:

            ## Initialize the prior
            pr = np.ones(pro.shape)

            ## Retrieve the length of the prototype UR and calculate
            ## the prior of generating a prototype UR of that length
            ln = self._vln(pro)
            pr *= self.geom.pmf(ln)
            pr *= Uniform.pmf(self.nsegs) ** ln

        ## Otherwise, calculate the probability of generating the specified
        ## contextual UR given the prototype UR
        else:

            ## Get indices of the prototype and contextual underlying forms
            ## given the underlying form matrix. Rows correspond to lexemes
            ## while columns correspond to contexts
            pid = self.cid[0]
            cid = self.cid

            ## Initialize the prior
            pr = np.ones(cxt.shape)

            ## Calculate the probability of each identity setting
            pr[cid] *= Bernoulli.pmf(idy[cid], self.al)

            ## If the identity setting is 1, then return whether the contextual
            ## underlying form matches the prototype underlying form. Otherwise,
            ## retrieve the edit distance from the prototype form to the given
            ## contextual underlying form
            p = self.dist.I.searchsorted(pro[pid])
            c = self.dist.I.searchsorted(cxt[cid])
            v = self.dist.prpmf((p, c)) ** (1 - idy[cid]) * (p == c) ** (idy[cid])
            pr[cid] *= v

        return pr

    def tppro(self, opro: np.ndarray, npro: np.ndarray):
        """Returns the transition probability between the current and new
        prototype UR for a given lexeme
        """

        ## Get the indices for each UR form hypothesis
        ofid = self.dist.I.searchsorted(opro)
        nfid = self.dist.I.searchsorted(npro)

        ## Get the transition probabilities from each index
        onpro = self.dist.tppmf((ofid, nfid))
        nopro = self.dist.tppmf((nfid, ofid))

        return onpro, nopro

    def tpcxt(
        self, ocxt: np.ndarray, oidy: np.ndarray, ncxt: np.ndarray, nidy: np.ndarray
    ):
        """Returns the transition probability between the current and new
        contextual UR
        """

        ## Initialize transition probability matrix
        oncxt = np.ones(self.cxt.shape)
        nocxt = np.ones(self.cxt.shape)

        ## Retrieve the indices of each lexeme for each context
        cid = self.cid

        ## Get the indices for each UR form hypothesis
        ofid = self.dist.I.searchsorted(ocxt[cid])
        nfid = self.dist.I.searchsorted(ncxt[cid])

        ## Get the transition probabilities from each index
        oncxt[cid] = self.dist.tppmf((ofid, nfid))
        nocxt[cid] = self.dist.tppmf((nfid, ofid))

        ## Get the indices for each parameter setting
        swid = (oidy[cid] != nidy[cid]).astype(int)

        ## Get the transition probabilities from each index
        oncxt[cid] *= Bernoulli.pmf(swid, self.bt)
        nocxt[cid] *= Bernoulli.pmf(swid, self.bt)

        return oncxt, nocxt

    ## *=*=*= SAMPLING METHODS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    def sample_pro(self, pro: np.ndarray, cxt: np.ndarray, idy: np.ndarray):
        """Samples a prototype underlying form for a given lexeme. Also returns
        the prior of the proposed UR
        """

        ## Retrieve the indices of the prototype underlying form string
        pid = self.dist.I.searchsorted(pro)

        ## Get the conditional probability of the proposal distribution given the string
        tppmf = self.dist.tppmf(pid)

        ## Sample a new underlying form given the conditional probabilities
        npro = self.dist.I[self.dist.rv(tppmf)]

        ## Get the identity parameters set to 1; these are the default parameters
        dft = np.nonzero(idy == 1)

        ## Update those marked with the default parameter to match the new proto UR
        ncxt = cxt.copy().astype(self.enc)
        ncxt[dft] = npro[dft[0]]

        ## Calculate the prior for the prototype UR and contextual UR
        nprpro = self.pr(pro=npro)
        nprcxt = self.pr(pro=npro, cxt=ncxt, idy=idy)

        return npro, ncxt, nprpro, nprcxt

    def sample_cxt(self, cxt: np.ndarray, idy: np.ndarray):
        """Samples a UR for a given lexical sequence given the proposal distribution.
        Also returns the prior of the proposed UR
        """

        ## Retrieve the indices of each lexeme for each context
        cid = self.cid

        ## Create copies of the identity and contextual underlying forms
        nidy = idy.copy()
        ncxt = cxt.copy()

        ## Sample the identity parameter for each contextual underlying form
        nidy[cid] = abs(idy[cid] - Bernoulli.rv(self.bt, idy[cid].shape))

        ## Depending on the sampled parameter, perfom a different outcome
        dft = np.nonzero(nidy == 1)
        gen = np.nonzero(nidy == 0)

        ## If it is the default parameter setting, set the contextual UR
        ## to be equal to the corrresponding prototype UR
        ncxt[dft] = self.pro[dft[0]]

        ## If it is not the default parameter, retrieve the indices of the
        ## contextual underlying form string
        xid = self.dist.I.searchsorted(cxt[gen])

        ## Get the conditional probability of the proposal distribution
        tppmf = self.dist.tppmf(xid)

        ## Given the indices, sample a new underlying form
        ncxt[gen] = self.dist.I[self.dist.rv(tppmf)]

        ## Calculate the prior for the contextual UR
        nprcxt = self.pr(pro=self.pro, cxt=ncxt, idy=nidy)

        return ncxt, nidy, nprcxt
