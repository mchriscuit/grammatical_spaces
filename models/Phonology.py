import numpy as np
import re
from itertools import permutations
from optim.Inventory import Inventory
from optim.Distributions import Binomial, Bernoulli, Uniform

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                PHONOLOGY (SPE vs. OT) DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class SPE(Inventory):
    """========== INITIALIZATION ======================================="""

    def __init__(
        self, mnames: list, mdefs: list, tokens: list, feats: list, configs: list
    ):
        self._rng = np.random.default_rng()

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        super().__init__(tokens, feats, configs)

        ## *=*=*= BASIC RULE INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._nms = len(mnames)
        self._mnames = np.array(mnames)
        self._mdefs = np.array(mdefs)

        ## *=*=*= INITIALIZE RULE HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._mhyps = []
        self._nmhyps = 0
        self._mhyp_idx = 0
        self.generate_hypotheses()

        ## *=*=*= INITIALIZE RULE CONFIGURATIONS *=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._mname2mconfig = {}
        self.configure_mappings()

        ## *=*=*= INITIALIZE RULE REGULAR EXPRESSIONS *=*=*=*=*=*=*=*=*=*=*=*
        self._mname2mregex = {}
        self.regex_mappings()

    """ ========== INSTANCE METHODS ===================================== """

    def update_mhyp_idx(self, idx: int):
        """Updates the rule hypothesis index"""
        self._mhyp_idx = idx

    def generate_hypotheses(self):
        """Given the set of rule (names), generate the space of
        possible mapping hypotheses. A random rule hypothesis is chosen
        as the initial rule hypothesis
        """
        for n in range(self.nms() + 1):
            self._mhyps += permutations(self.mnames(), n)
        self._nmhyps = len(self.mhyps())
        self._mhyp_idx = self._rng.choice(self.nmhyps())

    def configure_mappings(self):
        """Converts mappings into vector notation for process application"""

        def split_str_seq(str_seq_str_nat_classes: list):
            """Splits string sequence of string natural classes into a
            list of string natural classes
            """
            return str_seq_str_nat_classes.split(".")

        def split_str_nat_class(str_nat_classes: list):
            """Splits string sequence of features representing a natural class
            into a list of string features
            """
            return str_nat_classes.split(":")

        def generate_seq_config(seq_str_nat_classes: list):
            """Takes in a sequence of string natural class(es) and returns its
            feature configuration. If there is a ':' delimiter between
            features, then it is a multi-feature natural class. Word
            boundaries are represented as a vector of 0s
            """
            seq_config = []
            for str_nat_classes in seq_str_nat_classes:
                str_feats = split_str_nat_class(str_nat_classes)
                config = self.str_feats2config(str_feats)
                seq_config.append(config)
            return np.array(seq_config)

        for mname, rdef in zip(self.mnames(), self.mdefs()):
            src, tgt, cxt = map(lambda str_seq: split_str_seq(str_seq), rdef)
            idx = cxt.index("_")
            cxt[idx : idx + 1] = src
            tgt_config = generate_seq_config(tgt)
            cxt_config = generate_seq_config(cxt)
            self._mname2mconfig[mname] = (cxt_config, idx, tgt_config)

    def config_apply(self, seq_config: list):
        """Applies the phonological mapping given a sequence of feature
        configurations corresponding to some sequence of tokens
        """
        mhyp = self.get_current_mhyp()
        for mname in mhyp:
            seq_cxt_config, src_idx, tgt_config = self.mname2mconfig(mname)
            idxs = self.get_compatible_idxs(seq_config, seq_cxt_config)
            idxs += src_idx
            if len(idxs) != 0:
                seq_config = self.update_configs(seq_config, tgt_config, idxs)
        return seq_config

    def regex_mappings(self):
        """Converts mappings into regular expressions for process application"""
        for mname in self.mnames():
            cxt_config, src_idx, tgt_config = self._mname2mconfig[mname]
            cxt_token_configs = self.get_compatible_tokens(cxt_config)
            x = np.arange(len(cxt_token_configs))
            cxt_regex = [self.seq_config2tokens(c) for c in cxt_token_configs]
            cxt_regex = "".join(f"([{c}])" for c in cxt_regex)
            src_token_configs = cxt_token_configs[src_idx]
            y = np.arange(len(src_token_configs))
            src_regex = self.seq_config2tokens(src_token_configs)
            tgt_token_configs = self.update_config(src_token_configs, tgt_config, y)
            tgt_regex = self.seq_config2tokens(tgt_token_configs)
            tgt_regex = [
                "".join(f"\{i+1}" if i != src_idx else tgt for i in x)
                for tgt in tgt_regex
            ]
            tf_regex = {src: tgt for src, tgt in zip(src_regex, tgt_regex)}
            self._mname2mregex[mname] = (cxt_regex, tf_regex, src_idx + 1)

    def regex_apply(self, tokens: str):
        """Applies the phonological mapping given a sequence of tokens"""
        mhyp = self.get_current_mhyp()
        for mname in mhyp:
            cxt_regex, tf_regex, idx = self.mname2mregex(mname)
            tokens = re.sub(cxt_regex, lambda x: self.tf(x, tf_regex, idx), tokens)
        return tokens

    def tf(self, match: re.Match, tf_regex: dict, idx: int):
        """Returns a different regular expression depending on the match"""
        return match.expand(tf_regex[match.group(idx)])

    def compute_prior(self):
        """Computes the prior for the current rule hypothesis"""
        return 1

    """ ========== ACCESSORS ========================================= """

    def nms(self):
        """Returns the number of rules in the hypothesis space"""
        return self._nms

    def mnames(self):
        """Returns the rule names in the hypothesis space"""
        return self._mnames

    def mdefs(self):
        """Returns the dictionary from rule names to rule definitions"""
        return self._mdefs

    def mname2mconfig(self, mname: str):
        """Returns the rule configuration given the rule name"""
        return self._mname2mconfig[mname]

    def mname2mregex(self, mname: str):
        """Retrusn the rule regex given the rule name"""
        return self._mname2mregex[mname]

    def nmhyps(self):
        """Returns the number of rule hypotheses"""
        return self._nmhyps

    def mhyps(self):
        """Returns the rule hypotheses"""
        return self._mhyps

    def mhyp_idx(self):
        """Returns the index to the current rule hypothesis"""
        return self._mhyp_idx

    def get_current_mhyp(self):
        """Gets the current rule hypothesis"""
        return self._mhyps[self._mhyp_idx]


class OT:
    def __init__(self):
        pass
