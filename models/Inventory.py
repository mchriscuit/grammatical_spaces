import numpy as np

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                (FEATURE) INVENTORY DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class Inventory:
    """========== INITIALIZATION ======================================="""

    def __init__(self, segs: list, feats: list, configs: list):
        self._nsegs = len(segs)
        self._nfeats = len(feats)

        ## *=*=*= SEGMENTS AND FEATURES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._segs = np.array(segs)
        self._feats = np.array(feats)
        self._configs = np.array(configs).astype(int)

        ## *=*=*= INDEX DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._seg2id = {s: i for i, s in enumerate(self._segs)}
        self._feat2id = {f: i for i, f in enumerate(self._feats)}
        self._config2id = {tuple(c): i for i, c in enumerate(self._configs)}

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

    def segs2seq_config(self, segs: list):
        """Takes in a sequence of segments and returns a matrix of
        feature values
        """
        seq_config = [self.seg2config(seg) for seg in segs]
        seq_config = np.array(seq_config)
        return seq_config

    def seq_config2segs(self, seq_config: list):
        """Takes in a sequence of configurations and returns the sequence
        of segments corresponding to each one
        """
        segs = ""
        for config in seq_config:
            config = tuple(config)
            segs += self.config2seg(config)
        return segs

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

    def update_config(self, seq_seg_config: list, tgt_config: list, idxs: list):
        """Given a list of indices, a sequence of segment feature
        configurations and a target feature configuration, update the
        features with that configuration
        """
        m = ~np.isnan(tgt_config)
        for idx in idxs:
            seq_seg_config[np.newaxis, idx][m] = tgt_config[m]
        return seq_seg_config

    def is_compatible_seq_seg(self, seq_seg_config: list, seq_cxt_config: list):
        """Takes in a vector (or list) of feature configurations for a sequence
        of segments and a vector (or list) of the feature configurations for a
        context and returns a boolean corresponding to whether the sequence of
        segment configurations are compatible
        """
        m = ~np.isnan(seq_cxt_config)
        return np.allclose(seq_seg_config[m], seq_cxt_config[m])

    def get_compatible_idxs(self, seq_seg_config: list, seq_cxt_config: list):
        """Takes in a vector (or list) of feature configurations for a sequence
        of segments and a vector (or list) of the feature configurations for a
        context and returns a vector of indices where the contextual
        configuration is found in the segmental configuration sequence
        """
        compatible_idxs = []
        nsegs_seq, _ = seq_seg_config.shape
        nsegs_cxt, _ = seq_cxt_config.shape
        for i in range(nsegs_seq - nsegs_cxt + 1):
            sub_seq_seg_config = seq_seg_config[i : i + nsegs_cxt]
            if self.is_compatible_seq_seg(sub_seq_seg_config, seq_cxt_config):
                compatible_idxs.append(i)
        return np.array(compatible_idxs)

    """ ========== ACCESSORS ============================================ """

    def segs(self):
        """Returns the segments in the inventory"""
        return self._segs

    def feats(self):
        """Returns the feature names in the inventory"""
        return self._feats

    def configs(self):
        """Returns the feature configurations"""
        return self._configs

    def seg2config(self, seg: str):
        """Returns the feature value vector given a segment"""
        return self._configs[self._seg2id[seg]]

    def config2seg(self, config: list):
        """Returns the segment corresponding to the configuration"""
        config = tuple(config)
        return self._segs[self._config2id[config]]

    def feat2id(self, feat: str):
        """Returns the index of a feature"""
        return self._feat2id[feat]

    def nsegs(self):
        """Returns the number of segments in the inventory"""
        return self._nsegs

    def nfeats(self):
        """Returns the number of features in the inventory"""
        return self._nfeats


if __name__ == "__main__":
    print("Debugging mode...")
