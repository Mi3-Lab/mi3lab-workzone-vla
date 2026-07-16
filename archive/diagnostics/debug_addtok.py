import os
BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": f"{BASE}/.hf_cache"})
from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained(f"{BASE}/models/workzone-2b-stage7-1-hf")
tok = proc.tokenizer
print("vocab len antes:", len(tok), flush=True)
discrete_tokens = [f"<i{v}>" for v in range(4000)]
n = tok.add_tokens(discrete_tokens)
print("len(discrete_tokens):", len(discrete_tokens), flush=True)
print("num_new_tokens retornado:", n, flush=True)
print("vocab len depois:", len(tok), flush=True)
if n != len(discrete_tokens):
    missing = [t for t in discrete_tokens if t not in tok.get_added_vocab() and t not in tok.get_vocab()]
    present_already = [t for t in discrete_tokens if t in tok.get_vocab() and t not in tok.get_added_vocab()]
    print("tokens que ficaram fora do added_vocab:", len(missing), missing[:10], flush=True)
    print("tokens que ja existiam no vocab BASE (colisao):", len(present_already), present_already[:10], flush=True)
