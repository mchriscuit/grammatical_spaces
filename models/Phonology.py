import numpy as np
import re
import copy as cp
import itertools as it

from optim.Inventory import Inventory


class SPE(Inventory):
    """========== INITIALIZATION ================================================"""

    def __init__(self, mnames: np.ndarray, mdefs: np.ndarray, **kwargs):
        self._rng = np.random.default_rng()

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        super().__init__(**kwargs)

        ## *=*=*= BASIC RULE INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._nms = len(mnames)
        self._mnames = mnames
        self._mdefs = mdefs

        ## *=*=*= INITIALIZE RULE HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._mhyps = []
        self._nmhyps = 0
        self._mhyp_idx = 0
        self.generate_hypotheses()

        ## *=*=*= INITIALIZE RULE DICTIONARIES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._mhyp2idx = {mhyp: i for i, mhyp in enumerate(self._mhyps)}

        ## *=*=*= INITIALIZE RULE REGULAR EXPRESSIONS *=*=*=*=*=*=*=*=*=*=*=*
        self._mname2mregex = {}
        self.initialize_mappings()

    """ ========== OVERLOADING METHODS ========================================= """

    def __copy__(self):
        """Overloads the copy function"""
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__dict__.update(self.__dict__)
        return cp

    def __deepcopy__(self, memo: dict):
        """Returns a shallow copy of everything"""
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.__dict__.update(self.__dict__)
        return cp

    """ ========== INSTANCE METHODS ============================================ """

    def generate_hypotheses(self):
        """Given the set of rule (names), generate the space of
        possible mapping hypotheses. A random rule hypothesis is chosen
        as the initial rule hypothesis
        """
        for n in range(self.nms() + 1):
            self._mhyps += it.permutations(self.mnames(), n)
        self._nmhyps = len(self.mhyps())
        self._mhyp_idx = self._rng.choice(self.nmhyps())

    def initialize_mappings(self):
        """Converts mappings into regular expressions for process application"""

        def split_str_seq(str_seq_str_nat_classes: np.str_):
            """Splits string sequence of string natural classes into a
            list of string natural classes. If there are optional paradigms,
            split those into an internal list. Don't split it if it is bound
            within parentheses
            """
            return re.findall(
                r"(?:\.|^)(\(.*\)|.+?)(?:(?=\.|$))", str_seq_str_nat_classes
            )

        def split_opt_nat_class(str_nat_classes: np.str_):
            """Splits string sequence of features representing optional
            natural classes into a list of list of string features"""
            return [
                [split_str_nat_class(nc) for nc in snc.split(".")]
                for snc in re.sub(r"[\(\)]", "", str_nat_classes).split("|")
            ]

        def split_str_nat_class(str_nat_classes: np.str_):
            """Splits string sequence of features representing a natural class
            into a list of string features.
            """
            return str_nat_classes.split(":")

        def generate_seq_config(seq_str_nat_classes: list):
            """Takes in a sequence of string natural class(es) and returns its feature
            configuration. If there is a ':' delimiter between features, then it is a
            multi-feature natural class. If there is a '|' delimiter between features,
            then it is an alternative context. Alternative contexts are enclosed in
            parentheses '()' and are separated into their own sublists
            """
            seq_config = []
            for str_nat_classes in seq_str_nat_classes:
                if str_nat_classes == "_":
                    config = np.array([])
                else:
                    str_feats = (
                        split_opt_nat_class(str_nat_classes)
                        if re.search(r"\|", str_nat_classes)
                        else split_str_nat_class(str_nat_classes)
                    )
                    config = (
                        [np.array([self.seq2config(y) for y in x]) for x in str_feats]
                        if all(isinstance(x, list) for x in str_feats)
                        else self.seq2config(str_feats)
                    )
                seq_config.append(config)
            return seq_config

        """Main function definition"""
        for mname, rdef in zip(self.mnames(), self.mdefs()):

            ## Retrieve the source, target, and context
            src, tgt, cxt = map(lambda str_seq: split_str_seq(str_seq), rdef)
            src = split_str_nat_class(src[0])
            tgt = split_str_nat_class(tgt[0])

            ## Compute the configuration of the source and context
            src_config = self.seq2config(src)
            tgt_config = self.seq2config(tgt)

            ## Check whether the source or target is empty
            ins = np.isnan(src_config).all()
            dls = np.isnan(tgt_config).all()
            idx = cxt.index("_")

            ## Transform the source by the target value if it is a valid segment
            if not ins and not dls:
                src_configs = self.compatible(src_config)
                res_configs = self.update_config(src_configs, tgt_config)
            elif ins:
                src_configs = np.array([])
                res_configs = self.compatible(tgt_config)
            elif dls:
                src_configs = self.compatible(src_config)
                res_configs = np.array([])

            ## Get the array of tokens for the source and result
            src_tokens = self.configs2tokens(src_configs)
            res_tokens = self.configs2tokens(res_configs)

            ## Get the array of tokens for the context
            cxt_tokens = [
                self.configs2tokens(self.compatible(x))
                if isinstance(x, np.ndarray)
                else [[self.configs2tokens(self.compatible(z)) for z in y] for y in x]
                for x in generate_seq_config(cxt)
            ]

            ## Empty space string token for joining list elements
            i = ""

            ## Build the source regular expression from the tokens
            src_regex = f"([{i.join(src_tokens)}])".replace("[]", "")

            ## Build the context regular expression from the tokens
            cxt_regex = [
                f"([{i.join(y for y in x)}])"
                if all(isinstance(y, str) for y in x)
                else f"({'|'.join([''.join([f'[{i.join(z)}]' for z in y]) for y in x])})"
                for x in cxt_tokens
            ]

            ## Build the structure description regex from the combined tokens
            cxt_regex[idx] = src_regex
            cxt_regex = i.join(cxt_regex)

            ## Configure the result regex from the tokens
            x = np.arange(len(cxt_tokens))

            ## Depending on the number results, return a different regex
            if len(res_tokens) > 0:
                res_regex = [
                    i.join(rf"\{pos+1}" if pos != idx else res for pos in x)
                    for res in res_tokens
                ]
            else:
                res_regex = [
                    i.join(rf"\{pos+1}" for pos in x if pos != idx)
                    for src in src_tokens
                ]

            ## Finally, build the regular expression dictionary
            tf_regex = {}
            for src, res in it.zip_longest(src_tokens, res_regex):
                if src is None:
                    tf_regex[""] = res
                else:
                    tf_regex[src] = res

            ## Update the rule hypothesis
            self._mname2mregex[mname] = (cxt_regex, tf_regex, idx + 1)

    def apply(self, tokens: str, mhyp_idx=None):
        """Applies the phonological mapping given a sequence of tokens"""
        mhyp = self.get_current_mhyp() if mhyp_idx is None else self.get_mhyp(mhyp_idx)
        for mname in mhyp:
            cxt_regex, tf_regex, idx = self.mname2mregex(mname)
            tokens = re.sub(cxt_regex, lambda x: self.tf(x, tf_regex, idx), tokens)
        return tokens

    def tf(self, match: re.Match, tf_regex: dict, idx: int):
        """Returns a different regular expression depending on the match"""
        return match.expand(tf_regex.get(match.group(idx), tf_regex.get("")))

    """ ========== MUTATORS ==================================================== """

    def update_mhyp(self, mhyp: tuple):
        """Updates the rule hypothesis to the index corresponding to the name"""
        self._mhyp_idx = self._mhyp2idx[mhyp]

    def update_mhyp_idx(self, idx: int):
        """Updates the rule hypothesis index"""
        self._mhyp_idx = idx

    """ ========== ACCESSORS =================================================== """

    def nms(self):
        """Returns the number of rules in the hypothesis space"""
        return self._nms

    def mnames(self):
        """Returns the rule names in the hypothesis space"""
        return self._mnames

    def mdefs(self):
        """Returns the dictionary from rule names to rule definitions"""
        return self._mdefs

    def mname2mconfig(self, mname: np.str_):
        """Returns the rule configuration given the rule name"""
        return self._mname2mconfig[mname]

    def mname2mregex(self, mname: np.str_):
        """Retrusn the rule regex given the rule name"""
        return self._mname2mregex[mname]

    def nmhyps(self):
        """Returns the number of rule hypotheses"""
        return self._nmhyps

    def mhyps(self):
        """Returns the rule hypotheses"""
        return self._mhyps

    def get_current_mhyp_idx(self):
        """Returns the index to the current rule hypothesis"""
        return self._mhyp_idx

    def get_mhyp(self, mhyp_idx: int):
        """Returns the rule hypothesis given the index"""
        return self._mhyps[mhyp_idx]

    def get_current_mhyp(self):
        """Gets the current rule hypothesis"""
        return self.get_mhyp(self._mhyp_idx)
