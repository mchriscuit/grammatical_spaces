import numpy as np
import re
import itertools as it

from models.Inventory import Inventory
from models.Distributions import Geometric, Bernoulli, Uniform, Distance


class SPE(Inventory):
    """========== INITIALIZATION ================================================"""

    def __init__(self, ikwargs: dict, mnames: np.ndarray, mdefs: np.ndarray):
        self._rng = np.random.default_rng()

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(**ikwargs)

        ## *=*=*= BASIC RULE INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._mnames, self._nmnames = mnames, len(mnames) + 1
        self._mdefs = mdefs

        ## *=*=*= RULE HYPOTHESES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

        """Rule hypotheses ========================"""
        self._mhyps = np.array(
            [
                p
                for i in range(self.nmnames)
                for p in it.permutations(self.mnames.tolist(), i)
            ],
            dtype=object,
        )

        """Information and indexing ==============="""
        self._nmhyps = len(self.mhyps)
        self._mhidx = 0

        ## *=*=*= RULE DICTIONARY *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._mname2mregex = self.init_mappings(self.mnames, self.mdefs)

    """ ========== ACCESSORS ================================================== """

    @property
    def mnames(self):
        return self._mnames

    @property
    def nmnames(self):
        return self._nmnames

    @property
    def mdefs(self):
        return self._mdefs

    @property
    def mhyps(self):
        return self._mhyps

    @property
    def nmhyps(self):
        return self._nmhyps

    @property
    def mhidx(self):
        return self._mhidx

    @mhidx.setter
    def mhidx(self, idx: int):
        self._mhidx = idx

    @property
    def mname2mregex(self):
        return self._mname2mregex

    """ ========== INSTANCE METHODS ============================================ """

    def init_mappings(self, mnames: np.ndarray, mdefs: np.ndarray):
        """Converts mappings into regular expressions for process application"""
        mname2mregex = {}

        def split_str_seq(str_seq_str_nclasses: str):
            """Splits string sequence of string natural classes into a
            list of string natural classes. If there are optional paradigms,
            split those into an internal list. Don't split it if it is bound
            within parentheses
            """
            return re.findall(r"(?:\.|^)(\(.*\)|.+?)(?:(?=\.|$))", str_seq_str_nclasses)

        def split_opt_nclass(str_nclasses: str):
            """Splits string sequence of features representing optional
            natural classes into a list of list of string features"""
            return re.sub(r"[\(\)]", "", str_nclasses).split("|")

        def split_str_nclass(str_nclasses: str):
            """Splits string sequence of features representing a natural class
            into a list of string features.
            """
            return str_nclasses.split(":")

        def generate_cxt_regex(seq_str_nclasses: list):
            """Takes in a sequence of string natural class(es) and returns its feature
            configuration. If there is a ':' delimiter between features, then it is a
            multi-feature natural class. If there is a '|' delimiter between features,
            then it is an alternative context. Alternative contexts are enclosed in
            parentheses '()' and are separated into their own sublists
            """
            seq_cxt_regex = []

            ## Empty space string token for joining list elements
            i = ""

            ## The input is a list of contexts. As such, loop through each context
            for str_nclass in seq_str_nclasses:

                ## If the given natural class is '_', it is the source position; skip it
                if str_nclass == "_":
                    cxt_regex = ""

                ## Otherwise, it is a (n optional) natural class. We can check whether
                ## or not is it by looking at its type: if it a string that contains
                ## '|', it contains an optional position. Do some special
                ## operations on it...
                elif "|" in str_nclass:

                    ## Get the optional environment
                    opts = split_opt_nclass(str_nclass)
                    opts = [[o.split(":") for o in opt.split(".")] for opt in opts]

                    ## Get the compatible configurations
                    opt_configs = [
                        [self.nclass_to_config(o) for o in opt] for opt in opts
                    ]
                    opt_compat = [
                        [self.compatible_configs(o) for o in opt] for opt in opt_configs
                    ]

                    ## Get and join the associated segments
                    opt_tokens = [
                        [self.configs_to_tokens(o) for o in opt] for opt in opt_compat
                    ]
                    cxt_regex = [
                        i.join([f"[{i.join(list(set(o)))}]" for o in opt])
                        for opt in opt_tokens
                    ]
                    cxt_regex = f"({'|'.join(cxt_regex)})"

                ## Otherwise, it is a single natural class; perform as usual
                else:

                    ## Get the natural class
                    cxt = np.array(split_str_nclass(str_nclass))

                    ## Get the compatible configurations
                    cxt_config = self.nclass_to_config(cxt)
                    cxt_compat = self.compatible_configs(cxt_config)

                    ## Get and join the associated segments
                    cxt_tokens = self.configs_to_tokens(cxt_compat)
                    cxt_regex = f"([{''.join(list(set(cxt_tokens)))}])"

                ## Append to the list
                seq_cxt_regex.append(cxt_regex)

            return seq_cxt_regex

        """=============== Main function call ==================================="""
        for mname, rdef in zip(mnames, mdefs):
            """The source and target are always going to consist of a single row
            vector of features. Contexts, however, are a potential list of segments
            """

            ## Empty space string token for joining list elements
            i = ""

            ## Retrieve the source, target, and context
            src, tgt, cxt = map(lambda str_seq: split_str_seq(str_seq), rdef)
            src = split_str_nclass(src[0])
            tgt = split_str_nclass(tgt[0])
            idx = cxt.index("_")

            ## Generate the list regular expressions of joined tokens from the context
            seq_cxt_regex = generate_cxt_regex(cxt)

            ## Compute the configuration of the source and context
            src_config = self.nclass_to_config(src)
            tgt_config = self.nclass_to_config(tgt)

            ## Check whether the source or target is empty
            ins = np.isnan(src_config).all()
            dls = np.isnan(tgt_config).all()

            ## Transform the source by the target value if it is a valid segment
            if not ins and not dls:
                src_configs = self.compatible_configs(src_config)
                res_configs = self.update_config(src_configs, tgt_config)
            elif ins:
                src_configs = np.array([])
                res_configs = self.compatible_configs(tgt_config)
            elif dls:
                src_configs = self.compatible_configs(src_config)
                res_configs = np.array([])

            ## Get the array of tokens for the source and result
            src_tokens = self.configs_to_tokens(src_configs)
            res_tokens = self.configs_to_tokens(res_configs)

            ## Generate the regular expressions of the source segments
            src_regex = f"([{i.join(src_tokens)}])"
            src_regex = f"([*]?)" if src_regex == f"([])" else src_regex

            ## Generate the regular expression of the structural description
            seq_cxt_regex[idx] = src_regex
            sd_regex = re.compile(i.join(seq_cxt_regex))

            # Generate the regular expressions of the result segments
            x = np.arange(len(seq_cxt_regex))
            if len(res_tokens) > 0:
                res_regex = [
                    i.join(rf"\{w+1}" if w != idx else res for w in x)
                    for res in res_tokens.tolist()
                ]
            else:
                res_regex = [
                    i.join(rf"\{w+1}" for w in x if w != idx)
                    for src in src_tokens.tolist()
                ]
            res_regex = np.array(res_regex)

            ## Build the regular expression dictionary
            tf_regex = {}
            for src, res in it.zip_longest(src_tokens.tolist(), res_regex.tolist()):
                if src is None:
                    tf_regex[""] = res
                else:
                    tf_regex[src] = res

            ## Update the rule hypothesis
            mname2mregex[mname] = (sd_regex, tf_regex, idx)

        return mname2mregex

    def tf(self, match: re.Match, tf_regex: dict, idx: int):
        """Returns a different regular expression depending on the match"""
        return match.expand(tf_regex[match.groups("")[idx]])

    def apply(self, inputs: np.ndarray, mhidx: int = None):
        """Applies the phonological mapping given a sequence of tokens"""

        ## Check whether a mapping hypothesis is given; if not, use the
        ## current mapping hypothesis
        mhyp = self.mhyps[self.mhidx] if mhidx is None else self.mhyps[mhidx]

        ## Create a copy of the input array
        outputs = "|".join(inputs.tolist())

        ## For each input in the input array, apply each mapping hypothesis
        for mname in mhyp:
            sd_regex, tf_regex, idx = self.mname2mregex[mname]
            outputs = sd_regex.sub(lambda x: self.tf(x, tf_regex, idx), outputs)

        outputs = outputs.split("|")
        return outputs
