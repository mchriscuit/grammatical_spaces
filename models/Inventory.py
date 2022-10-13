import numpy as np
import re

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                (FEATURE) INVENTORY DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Inventory:
    """========== INITIALIZATION ======================================="""

    def __init__(self, tokens: list, feats: list, configs: list):
        self._rng = np.random.default_rng()

        ## *=*=*= HELPER FUNCTION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        def is_seg(token):
            return bool(re.match("\w", token))

        ## *=*=*= SEGMENTS AND FEATURES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._tokens = np.array(tokens)
        self._segs = self._tokens[list(map(is_seg, self._tokens))]
        self._feats = np.array(feats)
        self._tconfigs = np.array(configs).astype(int)
        self._sconfigs = self._tconfigs[list(map(is_seg, self._tokens))]

        ## *=*=*= SEGMENTS COUNTS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._ntokens = len(self._tokens)
        self._nsegs = len(self._segs)
        self._nfeats = len(self._feats)

        ## *=*=*= INDEX DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._token2id = {s: i for i, s in enumerate(self._tokens)}
        self._feat2id = {f: i for i, f in enumerate(self._feats)}
        self._tconfig2id = {tuple(c): i for i, c in enumerate(self._tconfigs)}

    """ ========== CLASS METHODS ======================================== """

    @classmethod
    def config_to_nan(cls, config: list):
        """Takes in a vector (or list) of feature values and
        returns a vector where all 0 features are set to NaN
        """
        config = np.array(config).astype("float")
        config[config == 0] = np.nan
        return config

    """ ========== INSTANCE METHODS ===================================== """

    def sample_sconfigs(self, n: int):
        """Samples n configs uniformly from the inventory"""
        return self._rng.choice(self.sconfigs(), n)

    def sample_segs(self, n: int):
        """Samples n segments uniformly from the inventory"""
        return self._rng.choice(self.segs(), n)

    def tokens2seq_config(self, tokens: list):
        """Takes in a sequence of tokens and returns a matrix of
        feature values
        """
        seq_config = [self.token2config(token) for token in tokens]
        seq_config = np.array(seq_config)
        return seq_config

    def seq_config2tokens(self, seq_config: list):
        """Takes in a sequence of configurations and returns the sequence
        of tokens corresponding to each one
        """
        tokens = ""
        for config in seq_config:
            config = tuple(config)
            tokens += self.config2token(config)
        return tokens

    def str_feats2config(self, str_feats: list):
        """Takes in a list of strings denoting the feature
        configuration for a natural class and returns a single vector
        with the denoted feature values.
        """
        config = np.zeros(self.nfeats())
        if str_feats == ["#"]:
            return config
        for str_feat in str_feats:
            typ, *feat = str_feat
            feat = "".join(feat)
            id = self.feat2id(feat)
            val = 1 if typ == "+" else -1
            config[id] = val
        return Inventory.config_to_nan(config)

    def update_config(self, seq_token_config: list, tgt_config: list, idxs: list):
        """Given a list of indices, a sequence of token feature
        configurations and a target feature configuration, update the
        features with that configuration
        """
        m = ~np.isnan(tgt_config)
        for idx in idxs:
            seq_token_config[np.newaxis, idx][m] = tgt_config[m]
        return seq_token_config

    def is_compatible_seq_token(self, seq_token_config: list, seq_cxt_config: list):
        """Takes in a vector (or list) of feature configurations for a sequence
        of tokens and a vector (or list) of the feature configurations for a
        context and returns a boolean corresponding to whether the sequence of
        token configurations are compatible
        """
        m = ~np.isnan(seq_cxt_config)
        return np.allclose(seq_token_config[m], seq_cxt_config[m])

    def get_compatible_idxs(self, seq_token_config: list, seq_cxt_config: list):
        """Takes in a vector (or list) of feature configurations for a sequence
        of tokens and a vector (or list) of the feature configurations for a
        context and returns a vector of indices where the contextual
        configuration is found in the token configuration sequence
        """
        compatible_idxs = []
        ntokens_seq, _ = seq_token_config.shape
        ntokens_cxt, _ = seq_cxt_config.shape
        for i in range(ntokens_seq - ntokens_cxt + 1):
            sub_seq_token_config = seq_token_config[i : i + ntokens_cxt]
            if self.is_compatible_seq_token(sub_seq_token_config, seq_cxt_config):
                compatible_idxs.append(i)
        return np.array(compatible_idxs)

    """ ========== ACCESSORS ============================================ """

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
        """Returns the feature configurations"""
        return self._tconfigs

    def sconfigs(self):
        """Returns the feature configurations"""
        return self._sconfigs

    def token2config(self, token: str):
        """Returns the feature value vector given a token"""
        return self._tconfigs[self._token2id[token]]

    def config2token(self, config: list):
        """Returns the token corresponding to the configuration"""
        config = tuple(config)
        if config in self._tconfig2id:
            return self._tokens[self._tconfig2id[config]]
        else:
            return "?"

    def feat2id(self, feat: str):
        """Returns the index of a feature"""
        return self._feat2id[feat]

    def ntokens(self):
        """Returns the number of tokens in the inventory"""
        return self._ntokens

    def nsegs(self):
        """Returns the number of segments in the inventory"""
        return self._nsegs

    def nfeats(self):
        """Returns the number of features in the inventory"""
        return self._nfeats
