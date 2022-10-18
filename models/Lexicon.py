import numpy as np
import re
from optim.Inventory import Inventory
from optim.Distributions import Binomial, Bernoulli, Uniform

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                LEXICON DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Lexicon(Inventory):
    """========== INITIALIZATION ======================================="""

    def __init__(
        self,
        lxs: list,
        clxs: list,
        ur_prior: tuple,
        ur_proposal: tuple,
        tokens: list,
        feats: list,
        configs: list,
    ):

        self._rng = np.random.default_rng()

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(tokens, feats, configs)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        """UR PRIOR"""
        self._pr_max_len = ur_prior["n"]
        self._pr_len = ur_prior["phi"]
        self._pr_del = ur_prior["alpha"]
        self._pr_swp = ur_prior["beta"]

        """UR PROPOSAL"""
        self._tp_max_len = ur_proposal["n"]
        self._tp_len = ur_proposal["chi"]
        self._tp_del = ur_proposal["delta"]
        self._tp_swp = ur_proposal["theta"]

        ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs = lxs
        self._clxs = clxs
        self._lbs = np.zeros(len(lxs))

        ## *=*=*= LEXICAL DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        """Lexeme to lexical sequence"""
        self._lx2clxs = {x: cxs for x, cxs in zip(self._lxs, self._clxs)}
        self._lx2clx2id = {
            x: {cx: i for i, cx in enumerate(self._lx2clxs[x])} for x in self._lxs
        }
        """Lexical sequence to lexeme"""
        self._clx2lxs = {
            cx: [x for x in self._lxs if x in cx] for cxs in self._clxs for cx in cxs
        }
        self._clx2lx2id = {
            cx: {x: i for i, x in enumerate(self._clx2lxs[cx])}
            for cx in self._clx2lxs.keys()
        }
        """Underlying form hypotheses"""
        self._lx2ur = {}
        self._lx2aln = {}
        self._lx2tp = {}
        self._lx2pr = {}

    """ ========== INSTANCE METHODS ===================================== """

    def initialize_urs(self, init_ur_hyp=None):
        """Initializes the UR hypotheses for the lexemes. If parameters
        are given, set those as the initial URs
        """
        if init_ur_hyp:
            init_lxs, init_clxs, init_urs, init_alns, lbs = init_ur_hyp
            assert set(self._lxs) == set(
                init_lxs
            ), "Specified lexemes and observed lexemes are not equal"
            assert all(
                set(self._lx2clxs[lx]) == set(init_clxs[i])
                for i, lx in enumerate(init_lxs)
            ), "Specified lexical contexts and observed lexical contexts not equal"
            for init_lx, init_ur, init_aln in zip(init_lxs, init_urs, init_alns):
                self._lbs = lbs
                init_ur = [self.tokens2seq_config(u) for u in init_ur]
                self._lx2ur[init_lx] = init_ur
                self._lx2aln[init_lx] = init_aln
                self._lx2tp[init_lx] = self.calculate_tp(init_ur, init_aln)
                self._lx2pr[init_lx] = self.calculate_pr(init_ur, init_aln)
        else:
            for lx in self.lxs():
                sampled_ur, sampled_aln, tp, pr = self.sample_ur(lx)
                self._lx2ur[lx] = sampled_ur
                self._lx2aln[lx] = sampled_aln
                self._lx2tp[lx] = tp
                self._lx2pr[lx] = pr

    def set_ur(self, lx, ur, aln, tp, pr):
        """Set the given lexeme to the given UR"""
        self._lx2ur[lx] = ur
        self._lx2aln[lx] = aln
        self._lx2tp[lx] = tp
        self._lx2pr[lx] = pr

    def sample_ur(self, lx, inplace=False):
        """Samples a UR for a given lexeme given the proposal distribution.
        Also returns the transition probability and prior of the proposed UR
        """
        nfeats = self.nfeats()
        for i, clx in enumerate(self.lx2clxs(lx)):
            if i == 0:
                pro_len = Binomial.rv(self._tp_max_len, self._tp_len)
                pro_ur = self.sample_sconfigs(pro_len)
                sampled_ur = []
                sampled_ur.append(pro_ur)
                pr = Binomial.pmf(pro_len, self._pr_max_len, self._pr_len)
                pr *= Uniform.pmf(self.segs()) ** pro_len
                tp = Binomial.pmf(pro_len, self._tp_max_len, self._tp_len)
                tp *= Uniform.pmf(self.segs()) ** pro_len
                sampled_aln = []
            else:
                deleted = Bernoulli.rv(self._tp_len, pro_len).astype(bool)
                ndeleted = np.sum(deleted == True)
                nkept = np.sum(deleted == False)
                cxt_ur = pro_ur[~deleted]
                cxt_aln = np.full(pro_len, "D")
                pr *= Bernoulli.pmf(1, self._pr_del) ** ndeleted
                pr *= Bernoulli.pmf(0, self._pr_del) ** nkept
                tp *= Bernoulli.pmf(1, self._tp_del) ** ndeleted
                tp *= Bernoulli.pmf(0, self._tp_del) ** nkept
                for j, cxt_idx in enumerate(np.argwhere(~deleted).squeeze(-1)):
                    swap = Bernoulli.rv(self._tp_swp, size=nfeats).astype(bool)
                    cxt_ur[j][swap] *= -1
                    cxt_aln[cxt_idx] = "N" if all(~swap) else "S"
                    nswapped = np.sum(swap == True)
                    nkept = np.sum(swap == False)
                    pr *= Bernoulli.pmf(1, self._pr_swp) ** nswapped
                    pr *= Bernoulli.pmf(0, self._pr_swp) ** nkept
                    tp *= Bernoulli.pmf(1, self._tp_swp) ** nswapped
                    tp *= Bernoulli.pmf(0, self._tp_swp) ** nkept
                sampled_ur.append(cxt_ur)
                sampled_aln.append(cxt_aln)
        if inplace:
            self.set_ur(lx, sampled_ur, sampled_aln, tp, pr)
        return sampled_ur, sampled_aln, tp, pr

    def calculate_pr(self, urs, aln):
        """Returns the prior probability of the given UR"""
        for i, ur in enumerate(urs):
            if i == 0:
                pro_ur = ur
                pro_len = len(ur)
                pr = Binomial.pmf(pro_len, self._pr_max_len, self._pr_len)
                pr *= Uniform.pmf(self.segs()) ** pro_len
            else:
                op = aln[i - 1]
                deleted = op == "D"
                pr *= Bernoulli.pmf(1, self._pr_del) ** np.sum(deleted)
                left = op != "D"
                pr *= Bernoulli.pmf(0, self._pr_del) ** np.sum(left)
                nswapped = np.sum(~(pro_ur[~deleted] == ur))
                pr *= Bernoulli.pmf(1, self._pr_swp) ** nswapped
                nkept = np.sum(pro_ur[~deleted] == ur)
                pr *= Bernoulli.pmf(0, self._pr_swp) ** nkept
        return pr

    def calculate_tp(self, urs, aln):
        """Returns the transition probability of the given UR"""
        for i, ur in enumerate(urs):
            if i == 0:
                pro_ur = ur
                pro_len = len(ur)
                tp = Binomial.pmf(pro_len, self._tp_max_len, self._tp_len)
                tp *= Uniform.pmf(self.segs()) ** pro_len
            else:
                op = aln[i - 1]
                deleted = op == "D"
                tp *= Bernoulli.pmf(1, self._tp_del) ** np.sum(deleted)
                left = op != "D"
                tp *= Bernoulli.pmf(0, self._tp_del) ** np.sum(left)
                nswapped = np.sum(~(pro_ur[~deleted] == ur))
                tp *= Bernoulli.pmf(1, self._tp_swp) ** nswapped
                nkept = np.sum(pro_ur[~deleted] == ur)
                tp *= Bernoulli.pmf(0, self._tp_swp) ** nkept
        return tp

    def add_padding(self, config, token="#"):
        """Pads the config with the specified token"""
        padding_config = self.token2config(token)
        return np.vstack((padding_config, config, padding_config))

    def rm_padding(self, config, token="#"):
        """Removes the padding from the given config"""
        padding_config = self.token2config(token)
        token_idx = ~(config[:, None] == padding_config).all(-1).squeeze()
        return config[token_idx, :]

    """ ========== ACCESSORS ============================================ """

    def lxs(self):
        """Returns the lexemes"""
        return self._lxs

    def clxs(self):
        """Returns the lexical sequences of each lexeme"""
        return self._clxs

    def lx2clxs(self, lx: str):
        """Returns a list of lexical contexts given a lexeme"""
        return self._lx2clxs[lx]

    def lx2clx2id(self, lx: str, clx: tuple):
        """Returns the index of a lexical context for a given lexeme"""
        return self._lx2clx2id[lx][clx]

    def lx2nclxs(self, lx: str):
        """Returns the number of lexical contexts given a lexeme"""
        return len(self._lx2clxs[lx])

    def clx2lxs(self, clx: tuple):
        """Returns a list of lexemes given a lexical context"""
        return self._clx2lxs[lx]

    def clx2lx2id(self, clx: tuple, lx: str):
        """Returns the index of a given lexeme for a given lexical context"""
        return self._clx2lx2id[clx][lx]

    def clx2nlxs(self, clx: tuple):
        """Returns the number of lexemes given a lexical context"""
        return len(self._clx2lxs[clx])

    def lx2lb(self, lx: str):
        """Returns the lower bound of a given lexeme"""
        return self._lbs[self._lx2id[lx]]

    def lx2ur(self, lx: str):
        """Returns the current UR hypothesis for a given lexeme"""
        return self._lx2ur[lx]

    def lx2aln(self, lx: str):
        """Returns the alignment for the current UR hypothesis"""
        return self._lx2aln[lx]

    def lx2tp(self, lx: str):
        """Returns the transition probability of the current UR hypothesis"""
        return self._lx2tp[lx]

    def lx2pr(self, lx: str):
        """Returns the prior probability of the current UR hypothesis"""
        return self._lx2pr[lx]

    def get_hyp(self, lx: str):
        """Returns the current UR hypothesis for a given lexeme"""
        return self.lx2ur(lx), self.lx2aln(lx), self.lx2tp(lx), self.lx2pr(lx)

    def get_ur(self, clx: tuple):
        """Returns the current UR for a given lexical context for a lexeme"""
        return np.vstack([self.lx2ur(lx)[self.lx2clx2id(lx, clx)] for lx in clx])
