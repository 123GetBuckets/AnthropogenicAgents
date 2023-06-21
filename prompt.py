#Bytes Path is just b'{FilePath}'
#context_p is a an array of values
from sre_constants import ANY_ALL
from common import GptParams
import llama_cpp
import sys 
from os import path
import ctypes

class Llama:
    def __init__(self, params: GptParams) ->None:
        self.params = params

		# runtime args
        self.input_consumed = 0
        self.n_past = 0
        self.n_session_consumed = 0
        self.first_antiprompt = []
        self.remaining_tokens = self.params.n_predict
        self.output_echo = self.params.input_echo
        self.multibyte_fix = []

		# model load
        self.lparams = llama_cpp.llama_context_default_params()
        self.lparams.n_ctx = self.params.n_ctx
        self.lparams.n_parts = self.params.n_parts
        self.lparams.seed = self.params.seed
        self.lparams.memory_f16 = self.params.memory_f16
        self.lparams.use_mlock = self.params.use_mlock
        self.lparams.use_mmap = self.params.use_mmap
        self.ctx = llama_cpp.llama_init_from_file(self.params.model.encode("utf8"), self.lparams)

        #internal context
        self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)

        #load session cache 
        self.session_tokens: list[llama_cpp.llama_token] = []
        if (len(self.params.path_session) > 0):
            print(f"Loading session from '{self.params.path_session}' ", file=sys.stderr)
            
            if (path.exists(self.params.path_session)):
                _session_tokens = (llama_cpp.llama_token * (self.params.n_ctx))()
                _n_token_count_out = llama_cpp.c_size_t()
                
                if(llama_cpp.llama_load_session_file(
                    self.ctx,
                    self.params.path_session.encode('utf8'),
                    _session_tokens,
                    self.params.n_ctx,
                    ctypes.byref(_n_token_count_out)
                )!=1):
                    print("Failed to Load Session File")
                    return
                _n_token_count_out = _n_token_count_out.value
                self.session_tokens = _session_tokens[:_n_token_count_out]
                print(f"Loaded Session of token ct {_n_token_count_out}", file=sys.stderr)
            else:
                print(f"Path does not exist, creating...", file=sys.stderr)

        #Tokenize Initial Prompt 
        self.embd = []
        self.embd_init = self.tokenize(self.params.prompt)

        if (len(self.embd_init) > self.n_ctx - 4):
            raise RuntimeError(f"error: prompt is too long ({len(self.embd_init)} tokens, max {self.params.n_ctx - 4})")

		# debug message about similarity of saved session, if applicable
        self.n_matching_session_tokens = 0
        if len(self.session_tokens) > 0:
            for id in self.session_tokens:
                if self.n_matching_session_tokens >= len(self.embd_init) or id != self.embd_init[self.n_matching_session_tokens]:
                    break
                self.n_matching_session_tokens += 1

            if self.n_matching_session_tokens >= len(self.embd_init):
                print(f"session file has exact match for prompt!")
            elif self.n_matching_session_tokens < (len(self.embd_init) / 2):
                print(f"warning: session file has low similarity to prompt ({self.n_matching_session_tokens} / {len(self.embd_init)} tokens); will mostly be reevaluated")
            else:
                print(f"session file matches {self.n_matching_session_tokens} / {len(self.embd_init)} tokens of prompt")

        self.need_to_save_session = len(self.params.path_session) > 0 and self.n_matching_session_tokens < (len(self.embd_init) * 3 / 4)
        if (self.params.n_keep < 0 or self.params.n_keep > len(self.embd_init) or self.params.instruct):
            self.params.n_keep = len(self.embd_init)

        print(f"""sampling: repeat_last_n = {self.params.repeat_last_n},\
        repeat_penalty = {self.params.repeat_penalty},\
        presence_penalty = {self.params.presence_penalty},\
        frequency_penalty = {self.params.frequency_penalty},\
        top_k = {self.params.top_k},\
        tfs_z = {self.params.tfs_z},\
        top_p = {self.params.top_p},\
        typical_p = {self.params.typical_p},\
        temp = {self.params.temp},\
        mirostat = {self.params.mirostat},\
        mirostat_lr = {self.params.mirostat_eta},\
        mirostat_ent = {self.params.mirostat_tau},\

        generate: n_ctx = {self.n_ctx},\
        n_batch = {self.params.n_batch},\
        n_predict = {self.params.n_predict},\
        n_keep = {self.params.n_keep},\
        session_tokens = {self.session_tokens}

        """, file=sys.stderr)








    def tokenize(self, prompt):
        l_tok = (llama_cpp.llama_token *((len(prompt) + 1) * 4))()
        n = llama_cpp.llama_tokenize(self.ctx, prompt.encode('utf8'), l_tok, len(l_tok), true)
        return l_tok[:n]
