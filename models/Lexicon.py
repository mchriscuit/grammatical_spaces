import numpy as np


class Lexicon:
    """========== INITIALIZATION ==================================================="""

    def __init__(self, lxs: np.ndarray, cxs: np.ndarray, ml: int, pad: str, *pdist):
        ## *=*=*= HELPER FUNCTIONS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._rng = np.random.default_rng()
        self._vln = np.vectorize(len)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._pad = pad
        self._snm = -1
        self._enc = "U" + f"{ml}"

        ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._lxs, self._nlxs = lxs, len(lxs)
        self._cxs, self._ncxs = cxs, len(cxs)

        ## *=*=*= DISTRIBUTIONS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._ulen, self._dfpr, self._dftp, self._pr, self._tp = pdist

        ## *=*=*= LEXICON HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._aid = [[self.lxs.searchsorted(c) for c in cx] for cx in self.cxs]
        self._bid = np.asarray([[lx in cx for cx in self.cxs] for lx in self.lxs])
        self._cid = np.nonzero(self.bid)
        pdim = (self.nlxs,)
        cdim = (self.nlxs, self.ncxs)

        """Pro underlying form (nlx, ) ============================================="""
        self._pro = np.full(pdim, "", dtype=self.enc)
        self._pro = self._pro  ## TODO: Initialize protytpe undelying forms

        """Cxt underlying form (nlx, ncx) =========================================="""
        self._cxt = np.full(cdim, "", dtype=self.enc)
        self._cxt = self._cxt  ## TODO: Initialize contextual underlying forms

        """Identity parameters (nlx, ncx) =========================================="""
        self._idy = np.full(cdim, self.snm, dtype=int)
        self._idy[self.cid] = 1  ## TODO: Initialize identity parameters

        """Underlying form priors =================================================="""
        self._prpro = self.pr(self.pro)
        self._prcxt = self.pr(self.pro, self.cxt, self.idy)

    """ ========== INSTANCE METHODS =============================================== """

    ## *=*=*= PROBABILITY METHODS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    def pr(self, pro: np.ndarray, cxt: np.ndarray = None, idy: np.ndarray = None):
        ## If no contextual underlying form vector is given, calculate the
        ## prior of the prototype underlying form
        if cxt is None:
            ## Initialize the prior
            pr = np.ones(pro.shape)

            ## Retrieve the length of the prototype UR and calculate
            ## the prior of generating a prototype UR of that length
            l = self.vln(pro)
            pr *= self._ulen.pmf(l)

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
            pr[cid] *= self._dfpr.pmf(idy[cid])

            ## If the identity setting is 1, then return whether the contextual
            ## underlying form matches the prototype underlying form. Otherwise,
            ## retrieve the edit distance from the prototype form to the given
            ## contextual underlying form
            p = self._pr.w.searchsorted(pro[pid])
            c = self._pr.w.searchsorted(cxt[cid])
            v = self._pr.pmf((p, c)) ** (1 - idy[cid]) * (p == c) ** (idy[cid])
            pr[cid] *= v

        return pr

    def tppro(self, opro: np.ndarray, npro: np.ndarray):
        """Returns the transition probability between the current and new
        prototype UR for a given lexeme
        """

        ## Get the indices for each UR form hypothesis
        ofid = self._tp.w.searchsorted(opro)
        nfid = self._tp.w.searchsorted(npro)

        ## Get the transition probabilities from each index
        onpro = self._tp.pmf((ofid, nfid))
        nopro = self._tp.pmf((nfid, ofid))

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
        ofid = self._tp.w.searchsorted(ocxt[cid])
        nfid = self._tp.w.searchsorted(ncxt[cid])

        ## Get the transition probabilities from each index
        oncxt[cid] = self._tp.pmf((ofid, nfid))
        nocxt[cid] = self._tp.pmf((nfid, ofid))

        ## Get the indices for each parameter setting
        swid = (oidy[cid] != nidy[cid]).astype(int)

        ## Get the transition probabilities from each index
        oncxt[cid] *= self._dftp.pmf(swid)
        nocxt[cid] *= self._dftp.pmf(swid)

        return oncxt, nocxt

    ## *=*=*= SAMPLING METHODS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

    def sample_pro(self, pro: np.ndarray, cxt: np.ndarray, idy: np.ndarray):
        """Samples a prototype underlying form for a given lexeme. Also returns
        the prior of the proposed UR
        """

        ## Retrieve the indices of the prototype underlying form string
        pid = self._tp.w.searchsorted(pro)

        ## Sample a new underlying form given the conditional probabilities
        npro = self._tp.w[self._tp.rvs(pid)]

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
        nidy[cid] = abs(idy[cid] - self._dftp.rvs(idy[cid].shape))

        ## Depending on the sampled parameter, perfom a different outcome
        dft = np.nonzero(nidy == 1)
        gen = np.nonzero(nidy == 0)

        ## If it is the default parameter setting, set the contextual UR
        ## to be equal to the corrresponding prototype UR
        ncxt[dft] = self.pro[dft[0]]

        ## If it is not the default parameter, retrieve the indices of the
        ## contextual underlying form string
        xid = self._tp.w.searchsorted(cxt[gen])

        ## Given the indices, sample a new underlying form
        ncxt[gen] = self._tp.w[self._tp.rvs(xid)]

        ## Calculate the prior for the contextual UR
        nprcxt = self.pr(pro=self.pro, cxt=ncxt, idy=nidy)

        return ncxt, nidy, nprcxt

    """ ========== ACCESSORS ====================================================== """

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
    def aid(self):
        return self._aid

    @property
    def bid(self):
        return self._bid

    @property
    def cid(self):
        return self._cid

    @property
    def vln(self):
        return self._vln

    def join(self, cxt: np.ndarray, pad: str = None):
        pad = self.pad if pad is None else pad
        aid = self.aid
        jxt = np.asarray([f"{pad}{''.join(cx[ai])}{pad}" for ai, cx in zip(aid, cxt.T)])
        return jxt
