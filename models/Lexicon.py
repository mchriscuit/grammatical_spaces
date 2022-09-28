import numpy as np
import re
from itertools import product
from models.Inventory import Inventory

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                LEXICON DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Lexicon(Inventory):
    """========== INITIALIZATION ======================================="""

    def __init__(
        self,
        lxs: list,
        clxs: list,
        urs: list,
        segs: list,
        feats: list,
        configs: list,
    ):
        self.rng = np.random.default_rng()

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(segs, feats, configs)

        ## *=*=*= BASIC LEXICAL INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lxs = lxs
        self._clxs = clxs
        self._urs = urs

        ## *=*=*= INDEX DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._lx2id = {lx: i for i, lx in enumerate(self._lxs)}
        self._ur2id = {ur: i for i, ur in enumerate(self._urs)}

        ## *=*=*= LEXICAL DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._lx2clxs = {x: cxs for x, cxs in zip(self._lxs, self._clxs)}
        self._lx2clx2id = {
            x: {cx: i for i, cx in enumerate(cxs)}
            for x, cxs in zip(self._lxs, self._clxs)
        }

        ## *=*=*= INITIALIZE LEXICON HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._lx2lhyps = {}
        self._lx2lhyp_idx = {}
        self.generate_hypotheses()

    """ ========== INSTANCE METHODS ===================================== """

    def update_lhyp_idx(self, lx: str, lhyp_idx: int):
        self._lx2lhyp_idx[lx] = lhyp_idx

    def generate_hypotheses(self):
        """Generate the space of possible URs for each lexeme"""
        for lx in self.lxs():
            urs = self.lx2urs(lx)
            nclxs = self.lx2nclxs(lx)
            lhyps = list(product(urs, repeat=nclxs))
            nlhyps = len(lhyps)
            self._lx2lhyps[lx] = lhyps
            self._lx2lhyp_idx[lx] = self.rng.choice(nlhyps)

    def compute_prior(self, lx):
        """Computes the prior for the current UR hypothesis for a given lexeme"""
        return 1

    def get_ur(self, clx: tuple):
        """Get the UR of a lexical sequence"""
        return "".join(self.get_current_lhyp(lx, clx) for lx in clx)

    def configure(self, segs, padding=True):
        """Configures the string. Optionally adds padding"""
        if padding:
            segs = f"#{segs}#"
        return self.segs2seq_config(segs)

    def deconfigure(self, seq_configs):
        """Converts a sequence of configurations to string form"""
        segs = self.seq_config2segs(seq_configs)
        return re.sub("#", "", segs)

    """ ========== ACCESSORS ============================================ """

    def lxs(self):
        """Returns the lexemes"""
        return self._lxs

    def clxs(self):
        """Returns the lexical sequences of each lexeme"""
        return self._clxs

    def urs(self):
        """Returns the underlying forms for each lexeme"""
        return self._urs

    def lx2clxs(self, lx: str):
        """Returns a list of lexical contexts given a lexeme"""
        return self._lx2clxs[lx]

    def lx2nclxs(self, lx: str):
        """Returns the number of lexical contexts given a lexeme"""
        return len(self._lx2clxs[lx])

    def lx2urs(self, lx: str):
        """Returns a list of URs for a given lexeme"""
        return self._urs[self._lx2id[lx]]

    def lhyps(self, lx: str):
        """Returns a list of UR hypotheses for a given lexeme"""
        return self._lx2lhyps[lx]

    def nlhyps(self, lx: str):
        """Returns the number of UR hypotheses for a given lexeme"""
        return len(self._lx2lhyps[lx])

    def lhyp_idx(self, lx: str):
        """Returns the index to the current UR hypothesis"""
        return self._lx2lhyp_idx[lx]

    def get_current_lhyp(self, lx: str, clx: tuple):
        """Returns the current UR hypothesis for a given
        lexeme for the given lexical sequence
        """
        lhyp_idx = self._lx2lhyp_idx[lx]
        id = self._lx2clx2id[lx][clx]
        return self._lx2lhyps[lx][lhyp_idx][id]
