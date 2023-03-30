import numpy as np
import re

from itertools import permutations, zip_longest
from models.Inventory import Inventory


class SPE(Inventory):
    """========== INITIALIZATION ================================================"""

    def __init__(self, ikw: dict, mnms: np.ndarray, mdfs: np.ndarray):

        ## *=*=*= INHERITENCE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        super().__init__(**ikw)
        self._rng = np.random.default_rng()
        self._vln = np.vectorize(len)

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._bnd = "."

        ## *=*=*= BASIC RULE INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._mnms, self._nmns = mnms, len(mnms) + 1
        self._mdfs = mdfs

        """Rule hypotheses ========================"""
        self._mhys = [m for i in range(self.nmns) for m in permutations(self.mnms, i)]
        self._mhys, self._nmhs = np.array(self.mhys, dtype=object), len(self.mhys)
        self._mhid = np.zeros((1,)).astype(int)

        """Rule mappings =========================="""
        self._mn2rx = self.init_mappings(self.mnms, self.mdfs)

        """Pre-processed arrays ==================="""
        self._ml = None
        self._fm = None
        self._fm2ex = None

    """ ========== ACCESSORS ================================================== """

    @property
    def ml(self):
        return self._ml

    @property
    def bnd(self):
        return self._bnd

    @property
    def mnms(self):
        return self._mnms

    @property
    def nmns(self):
        return self._nmns

    @property
    def mdfs(self):
        return self._mdfs

    @property
    def mhys(self):
        return self._mhys

    @property
    def nmhs(self):
        return self._nmhs

    @property
    def mhid(self):
        return self._mhid

    @mhid.setter
    def mhid(self, id: int):
        self._mhid[0] = id

    @property
    def mn2rx(self):
        return self._mn2rx

    @property
    def fm(self):
        return self._fm

    @property
    def fm2ex(self):
        return self._fm2ex

    @property
    def vln(self):
        return self._vln

    """ ========== INSTANCE METHODS ============================================ """

    def preprocess_tf(self, forms: np.ndarray):
        """Given the current space of rule mappings and space of possible (padded)
        inputs, returns a cached matrix of input x mhyp outputs
        """

        ## Update class variables
        self._fm = forms
        self._ml = self.vln(forms).max()

        ## Retrieve the dimensions of the output matrix
        nfms = forms.size
        nmhs = self.nmhs

        ## Initialize the empty transformation cache
        mdim = (nfms, nmhs)
        fm2ex = np.full(mdim, "", dtype="U" + f"{self.ml}")

        ## For each rule hypothesis, generate the predictions for each input
        fm2ex = self.tfapply(forms=forms, mhid=np.arange(nmhs))

        ## Update the class variable
        self._fm2ex = fm2ex.copy()

    def init_mappings(self, mnms: np.ndarray, mdfs: np.ndarray):
        """Converts mappings into regular expressions for process application"""
        mnm2reg = {}

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
        for mnm, mdf in zip(mnms.tolist(), mdfs.tolist()):
            """The source and target are always going to consist of a single row
            vector of features. Contexts, however, are a potential list of segments
            """

            ## Empty space string token for joining list elements
            i = ""

            ## Retrieve the source, target, and context
            src, tgt, cxt = map(lambda str_seq: split_str_seq(str_seq), mdf)
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
            for src, res in zip_longest(src_tokens.tolist(), res_regex.tolist()):
                if src is None:
                    tf_regex[""] = res
                else:
                    tf_regex[src] = res

            ## Update the rule hypothesis
            mnm2reg[mnm] = (sd_regex, tf_regex, idx)

        return mnm2reg

    def tf(self, match: re.Match, tf_regex: dict, idx: int):
        """Returns a different regular expression depending on the match"""
        return match.expand(tf_regex[match.groups("")[idx]])

    def tfapply(self, forms: np.ndarray, mhid: np.ndarray = None):
        """Applies the phonological mapping given a sequence of padded tokens using a dict"""

        ## Check whether a mapping hypothesis is given; if not, use the
        ## current mapping hypothesis
        if mhid is None:
            mhys = self.mhys[self.mhid].tolist()
            mdim = (len(forms), 1)
        else:
            mhys = self.mhys[mhid].tolist()
            mdim = (len(forms), len(mhid))

        ## Create an numpy array corresponding to the expected string
        exs = np.empty(mdim, dtype="U" + f"{3 * self.ml}")

        ## For each indexed hypothesis, for each input
        ## in the input array, apply each mapping hypothesis
        for i, mhy in enumerate(mhys):
            ex = self.bnd.join(forms.tolist())
            for mn in mhy:
                sd, tf, idx = self.mn2rx[mn]
                ex = sd.sub(lambda x: self.tf(x, tf, idx), ex)
            exs[:, i] = np.array(ex.split(self.bnd))

        return exs

    def chapply(self, forms: np.ndarray, mhid: np.ndarray = None):
        """Applies the phonological mapping given a sequence of padded tokens using a cache"""

        ## Check whether a mapping hypothesis is given; if not, use the
        ## current mapping hypothesis
        mhid = self.mhid if mhid is None else mhid

        ## Get the indices for the forms
        ufid = self.fm.searchsorted(forms)

        ## Get the outputs for each index: First, retrieve the rows of each indexed form
        ## Next, get the column of expected outputs for each indexed rule hypothesis
        exs = self.fm2ex[ufid][:, mhid]

        return exs

    def apply(self, forms: np.ndarray, mhid: np.ndarray = None):
        if self.vln(forms).max() > self.ml:
            return self.tfapply(forms=forms, mhid=mhid)
        else:
            return self.chapply(forms=forms, mhid=mhid)
