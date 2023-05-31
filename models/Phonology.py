import numpy as np
import re
import itertools as it


class SPE:
    """========== INITIALIZATION ==================================================="""

    def __init__(self, mnms: np.ndarray, mdfs: np.ndarray, fms: np.ndarray, inv):
        ## *=*=*= HELPER FUNCTIONS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._rng = np.random.default_rng()
        self._vln = np.vectorize(len)
        self._inv = inv

        ## *=*=*= HYPERPARAMETERS *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        self._bnd = "."

        ## *=*=*= BASIC RULE INFORMATION *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        self._mnms, self._nmns = mnms, len(mnms) + 1
        self._mdfs = mdfs

        """Rule hypotheses ========================================================="""
        ## self._mhs = [m for i in range(self.nmns) for m in it.permutations(self.mnms, i)] 
        ## REMOVE THIS LATER
        self._mhs = []
        for i in range(self.nmns):
            for j in it.permutations(range(2), i):
                if len(j) > 1:
                    self._mhs.append(tuple([self.mnms[j[0] * 2]] + [self.mnms[j[0] * 2 + 1]] + [self.mnms[j[1]* 2] ] + [self.mnms[j[1] * 2 + 1]]))
                elif len(j) == 1:
                    self._mhs.append(tuple([self.mnms[j[0] * 2]] + [self.mnms[j[0] * 2 + 1]]))
                else:
                    self._mhs.append(tuple())
        self._mhs, self._nhs = np.asarray(self.mhs, dtype=object), len(self.mhs)
        self._mhi = np.asarray([0])

        """Rule mappings ==========================================================="""
        self._mnm2rgx = self.init_mappings(self.mnms, self.mdfs)

        """Pre-processed arrays ===================================================="""
        self._fms, self._nfms = fms, len(fms)
        self._mxl = self.vln(self.fms).max()
        self._fms2exs = self.preprocess_tf()

    """ ========== INSTANCE METHODS ================================================ """

    def preprocess_tf(self):
        """Given the current space of rule mappings and space of possible (padded)
        inputs, returns a cached matrix of ``fms`` x ``mhs`` outputs
        """

        ## For each rule hypothesis, generate the predictions for each input
        fms2exs = self.tfapply(fms=self.fms, mhi=np.arange(self.nhs))

        return fms2exs

    def tfapply(self, fms: np.ndarray, mhi: np.ndarray = None):
        """Applies the phonological mapping given a sequence of padded tokens using a dict"""

        ## Check whether a mapping hypothesis is given
        mhi = self.mhi if mhi is None else mhi

        ## Retrieve the dimensions of the output matrix
        mhys = self.mhs[mhi].tolist()
        mdim = (fms.size, mhi.size)

        ## Create an numpy array corresponding to the expected string
        exs = np.empty(mdim, dtype="U" + f"{3 * self.mxl}")
        jnd = self.bnd.join(fms.tolist())

        ## For each indexed hypothesis, for each input
        ## in the input array, apply each mapping hypothesis
        for i, mhy in enumerate(mhys):
            out = jnd
            for mnm in mhy:
                run = True
                while run:
                    old = out
                    sd, tf = self.mnm2rgx[mnm]
                    out = sd.sub(lambda x: tf[x.groups("")[0]], out)
                    run = re.search(sd, out)
            exs[:, i] = np.asarray(out.split(self.bnd))
        return exs

    def chapply(self, fms: np.ndarray, mhi: np.ndarray = None):
        """Applies the phonological mapping given a sequence of padded tokens using a cache"""

        ## Check whether a mapping hypothesis is given
        mhi = self.mhi if mhi is None else mhi

        ## Get the indices for the forms
        ufid = self.fms.searchsorted(fms)

        ## Get the outputs for each index: First, retrieve the rows of each indexed form
        ## Next, get the column of expected outputs for each indexed rule hypothesis
        exs = self.fms2exs[ufid][:, mhi]

        return exs

    def apply(self, fms: np.ndarray, mhi: np.ndarray = None):
        if self.vln(fms).max() > self.mxl:
            return self.tfapply(fms=fms, mhi=mhi)
        else:
            return self.chapply(fms=fms, mhi=mhi)

    def init_mappings(self, mnms: np.ndarray, mdfs: np.ndarray):
        """Converts mappings into regular expressions for process application"""
        mnm2rgx = {}

        def split_str_seq(str_seq_str_nclasses: str):
            """Splits string sequence of string natural classes into a list of string
            natural classes. If there are optional paradigms, split those into an internal
            list. Don't split it if it is bound within parentheses
            """
            return re.findall(r"(?:\.|^)(\(.*\)|.+?)(?:(?=\.|$))", str_seq_str_nclasses)

        def split_opt_nclass(str_nclasses: str):
            """Splits string sequence of features representing optional natural classes into
            a list of list of string features
            """
            return re.sub(r"[\(\)]", "", str_nclasses).split("|")

        def split_str_nclass(str_nclasses: str):
            """Splits string sequence of features representing a natural class into a list
            of string features
            """
            return str_nclasses.split(":")

        def gen_cxt_rgx(seq_str_nclasses: list):
            """Takes in a sequence of string natural class(es) and returns its feature
            configuration. If there is a ':' delimiter between features, then it is a
            multi-feature natural class. If there is a '|' delimiter between features,
            then it is an alternative context. Alternative contexts are enclosed in
            parentheses '()' and are separated into their own sublists
            """
            seq_cxt_rgx = []

            ## Empty space string token for joining list elements
            i = ""

            ## The input is a list of contexts. As such, loop through each context
            for str_nclass in seq_str_nclasses:
                ## If the given natural class is '_', it is the source position; skip it
                if str_nclass == "_":
                    cxt_rgx = ""

                ## Otherwise, it is a (n optional) natural class. We can check whether
                ## or not is it by looking at its type: if it a string that contains
                ## '|', it contains an optional position. Do some special operations on it...
                elif "|" in str_nclass:
                    ## Get the optional environment
                    opts = split_opt_nclass(str_nclass)
                    opts = [[o.split(":") for o in opt.split(".")] for opt in opts]

                    ## Get the compatible configurations
                    opt_cfgs = [
                        [self.inv.nclass_to_config(o) for o in op] for op in opts
                    ]
                    opt_cmp = [
                        [self.inv.compatible_configs(o) for o in op] for op in opt_cfgs
                    ]

                    ## Get and join the associated segments
                    opt_tok = [
                        [self.inv.configs_to_tokens(o) for o in op] for op in opt_cmp
                    ]
                    cxt_rgx = [
                        i.join([f"[{i.join(list(set(o)))}]" for o in op])
                        for op in opt_tok
                    ]
                    cxt_rgx = f"({'|'.join(cxt_rgx)})"

                ## Otherwise, it is a single natural class; perform as usual
                else:
                    ## Get the natural class
                    cxt = np.asarray(split_str_nclass(str_nclass))

                    ## Get the compatible configurations
                    cxt_cfg = self.inv.nclass_to_config(cxt)
                    cxt_cmp = self.inv.compatible_configs(cxt_cfg)

                    ## Get and join the associated segments
                    cxt_toks = self.inv.configs_to_tokens(cxt_cmp)
                    cxt_rgx = f"[{''.join(list(set(cxt_toks)))}]"

                ## Append to the list
                seq_cxt_rgx.append(cxt_rgx)

            return seq_cxt_rgx

        """=============== Main function call ======================================"""
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
            seq_cxt_regex = gen_cxt_rgx(cxt)

            ## Compute the configuration of the source and context
            src_config = self.inv.nclass_to_config(src)
            tgt_config = self.inv.nclass_to_config(tgt)

            ## Check whether the source or target is empty
            ins = np.isnan(src_config).all()
            dls = np.isnan(tgt_config).all()

            ## Transform the source by the target value if it is a valid segment
            if not ins and not dls:
                src_configs = self.inv.compatible_configs(src_config)
                res_configs = self.inv.update_config(src_configs, tgt_config)
            elif ins:
                src_configs = np.asarray([])
                res_configs = self.inv.compatible_configs(tgt_config)
            elif dls:
                src_configs = self.inv.compatible_configs(src_config)
                res_configs = np.asarray([])

            ## Get the array of tokens for the source and result
            fsrc_tokens = self.inv.configs_to_tokens(src_configs)
            fres_tokens = self.inv.configs_to_tokens(res_configs)

            ## Remove vacuous pairs from the array
            src_tokens = np.asarray(
                [
                    src
                    for src, res in it.zip_longest(fsrc_tokens, fres_tokens)
                    if src != res and src is not None
                ]
            )
            res_tokens = np.asarray(
                [
                    res
                    for src, res in it.zip_longest(fsrc_tokens, fres_tokens)
                    if src != res
                ]
            )

            ## Generate the regular expressions of the source segments
            src_regex = f"([{i.join(src_tokens)}])"
            src_regex = f"([*]?)" if src_regex == f"([])" else src_regex

            ## Generate the regular expression of the structural description
            seq_cxt_regex[idx] = src_regex

            ## Fix the context such that the lookaheads before the idx are
            ## converted into lookbehinds
            seq_cxt_regex = [
                f"(?<={cxt}" if x == 0 and x < idx else 
                f"(?={cxt}" if x == idx + 1 and x + 1 < len(seq_cxt_regex) else
                f"(?<={cxt})" if x == 0 and x + 1 == idx else 
                f"(?={cxt})" if x == idx + 1 and x + 1 == len(seq_cxt_regex) else
                f"{cxt})" if x + 1 == idx or x + 1 == len(seq_cxt_regex) and x != idx else 
                cxt for x, cxt in enumerate(seq_cxt_regex)
            ]

            ## Generate the regular expression for the structural description
            sd_regex = re.compile(i.join(seq_cxt_regex))

            # Generate the regular expressions of the result segments
            res_regex = np.asarray(res_tokens)

            ## Build the regular expression dictionary
            tf_regex = {}
            for src, res in it.zip_longest(src_tokens.tolist(), res_regex.tolist()):
                if src is None:
                    tf_regex[""] = res
                if src != res:
                    tf_regex[src] = res

            ## Update the rule hypothesis
            mnm2rgx[mnm] = (sd_regex, tf_regex)

        return mnm2rgx

    """ ========== ACCESSORS ====================================================== """

    @property
    def vln(self):
        return self._vln

    @property
    def mxl(self):
        return self._mxl

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
    def mhs(self):
        return self._mhs

    @property
    def nhs(self):
        return self._nhs

    @property
    def mhi(self):
        return self._mhi

    @mhi.setter
    def mhi(self, id: int):
        self._mhi[0] = id

    @property
    def mnm2rgx(self):
        return self._mnm2rgx

    @property
    def fms(self):
        return self._fms

    @property
    def nfms(self):
        return self._nfms

    @property
    def fms2exs(self):
        return self._fms2exs

    @property
    def inv(self):
        return self._inv
