import numpy as np
import re
import copy as cp


class Inventory:
    """========== INITIALIZATION ================================================="""

    def __init__(self, tokens: np.ndarray, feats: np.ndarray, configs: np.ndarray):
        self._rng = np.random.default_rng()

        ## *=*=*= HELPER FUNCTION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        def is_seg(token):
            return bool(re.match("\w", token))

        ## *=*=*= SEGMENTS AND FEATURES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._tokens = tokens
        self._segs = self._tokens[list(map(is_seg, self._tokens))]
        self._feats = feats
        self._tconfigs = configs
        self._sconfigs = self._tconfigs[list(map(is_seg, self._tokens))]

        ## *=*=*= SEGMENTS COUNTS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._ntokens = len(self._tokens)
        self._nsegs = len(self._segs)
        self._nfeats = len(self._feats)

        ## *=*=*= INDEX DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._token2id = {s: i for i, s in enumerate(self._tokens)}
        self._feat2id = {f: i for i, f in enumerate(self._feats)}
        self._tconfig2id = {tuple(c): i for i, c in enumerate(self._tconfigs)}

    """ ========== STATIC METHODS =============================================== """

    @staticmethod
    def config_to_nan(config: np.ndarray):
        """Takes in a vector (or list) of feature values and returns an array where
        all 0 features are set to NaN
        """
        config = config.astype("float")
        config[config == 0] = np.nan
        return config

    """ ========== INSTANCE METHODS =============================================== """

    def sample_segs(self, n: int):
        """Samples n segments uniformly from the inventory"""
        return self._rng.choice(self.segs(), n)

    def update_config(self, src_config: np.ndarray, tgt_config: np.ndarray):
        """Update the source features with the target configuration"""
        out_config = cp.deepcopy(src_config)
        m = ~np.isnan(tgt_config).squeeze()
        out_config[..., m] = tgt_config[..., m].astype(int)
        return out_config

    """ ========== BASIC ACCESSORS ================================================= """

    def tokens(self):
        """Returns the tokens in the inventory"""
        return self._tokens

    def segs(self):
        """Returns the segments in the inventory"""
        return self._segs

    def feats(self):
        """Returns the feature names in the inventory"""
        return self._feats

    def tconfigs(self):
        """Returns the token feature configurations"""
        return self._tconfigs

    def sconfigs(self):
        """Returns the segment feature configurations"""
        return self._sconfigs

    def ntokens(self):
        """Returns the number of tokens in the inventory"""
        return self._ntokens

    def nsegs(self):
        """Returns the number of segments in the inventory"""
        return self._nsegs

    def nfeats(self):
        """Returns the number of features in the inventory"""
        return self._nfeats

    def token2id(self, token: np.str_):
        """Returns the index of a token"""
        return self._token2id[token]

    def feat2id(self, feat: np.str_):
        """Returns the index of a feature"""
        return self._feat2id[feat]

    def config2id(self, config: np.ndarray):
        """Returns the index of a configuration"""
        config = tuple(config)
        return self._tconfig2id[config]

    """ ========== GENERATIVE ACCESSORS =========================================== """

    def tokens2configs(self, tokens: np.ndarray):
        """Returns an array of configurations given an array of tokens"""
        return self._tconfigs[[self.token2id(token) for token in tokens]]

    def configs2tokens(self, configs: np.ndarray):
        """Returns the tokens given an array of configurations"""
        return self._tokens[[self.config2id(config) for config in configs]]

    def seq2config(self, seq: np.ndarray):
        """Takes in a list of strings denoting the features for a natural class
        or a list of tokens and returns a single configuration with the
        provided feature values
        """

        ## Check whether the input is a sequence of tokens
        if np.isin(seq, self.tokens()).all():
            config = self.intersect(seq)
            return config

        ## Otherwise, assume that it is a sequence of features
        config = np.full((1, self.nfeats()), np.nan)
        for feat in seq:
            typ, *feat = feat
            feat = "".join(feat)
            id = self.feat2id(feat)
            val = 1 if typ == "+" else -1
            config[:, id] = val

        return config

    def compatible(self, seq_config: np.ndarray):
        """Takes in a vector of feature configurations and returns all
        token configurations compatible with that configuration. Compatible
        tokens are determined by seeing whether all of the non-NaN values are
        matched in the segment. An empty array is returned if there are no
        compatible matches in the configuration
        """

        ## Retrieve the token configs
        tconfigs = self.tconfigs()

        ## Loop through each configuration in the sequence
        segs = []
        for config in seq_config:

            ## If the configs given are all nan, then it is a word boundary
            config = np.zeros(config.shape) if np.isnan(config).all() else config

            ## Retrieve all the non-nan cells and compare
            m = ~np.isnan(config)
            compatible_configs = (tconfigs[:, m] == config[m]).all(axis=1)
            segs.append(tconfigs[compatible_configs])

        return np.vstack(segs) if len(segs) > 0 else np.array(segs)

    def intersect(self, tokens: np.ndarray):
        """Returns the maximal natural class configuration consisting of the given
        tokens and nothing else. Returns an empty array if no compatible natural
        class is found. Works for both special characters and segments
        """

        ## Get all the tokens and other segments
        segments = self.segs()
        osegments = segments[~np.isin(segments, tokens)]

        ## Get the token and alternative configs
        tconfigs = self.tokens2configs(tokens)
        oconfigs = self.tokens2configs(osegments)

        ## Initialize an empty configuration
        nfeats = self.nfeats()
        cconfig = np.zeros((1, nfeats))

        ## Get all the indices where the feature values overlap
        cidx = (tconfigs == tconfigs[0, :].reshape(1, -1)).all(axis=0)

        ## Check to make sure that the natural class only refers to the given tokens
        assert (
            ~(tconfigs[0, cidx] == oconfigs[:, cidx]).all(axis=1).any()
        ), f"{tokens} cannot create a unique natural class!"

        ## Otherwise, update the config and return it
        cconfig[:, cidx] = tconfigs[0, cidx]

        return Inventory.config_to_nan(cconfig)
