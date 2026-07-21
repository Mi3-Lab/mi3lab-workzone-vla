"""Quantiza a torre de texto do Cosmos3-Edge (INT4, torchao, arquitetura
NATIVA) e SALVA o resultado em disco, alem de validar nas 5 perguntas padrao
(GATE/EGO/ACTIVE/SIGN/DESC) comparando com os resultados conhecidos de hoje
(job 194667, modelo BF16 original).
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import cosmos_framework.inference.inference as inf_mod

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
OUT_DIR = f"{BASE}/models/cosmos3edge-lm-int4-torchao"

_orig_create = inf_mod.OmniInference.create.__func__


def real_serialized_bytes(m):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name)
        size = os.path.getsize(f.name)
        os.unlink(f.name)
    return size


def _patched_create(cls, setup_args, /):
    pipe = _orig_create(cls, setup_args)
    model = pipe.model
    lm = model.net.language_model

    size_before = real_serialized_bytes(lm)
    print(f"[quant] tamanho ANTES (serializado real): {size_before / 1e9:.3f} GB", flush=True)

    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
    qconfig = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)

    def _filter_fn(module, fqn):
        return isinstance(module, torch.nn.Linear) and "moe_gen" not in fqn

    n_eligible = sum(1 for name, mod in lm.named_modules() if _filter_fn(mod, name))
    print(f"[quant] {n_eligible} camadas nn.Linear elegiveis (fora _moe_gen)", flush=True)

    quantize_(lm, qconfig, filter_fn=_filter_fn)
    print("[quant] quantize_() concluido", flush=True)

    size_after = real_serialized_bytes(lm)
    print(f"[quant] tamanho DEPOIS (serializado real): {size_after / 1e9:.3f} GB "
         f"(reducao: {100 * (1 - size_after / size_before):.1f}%)", flush=True)

    print(f"\n[save] salvando state_dict quantizado em {OUT_DIR}/lm_int4_state_dict.pt ...", flush=True)
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(lm.state_dict(), f"{OUT_DIR}/lm_int4_state_dict.pt")
    with open(f"{OUT_DIR}/README.txt", "w") as f:
        f.write(
            "Torre de texto do Cosmos3-Edge (Nemotron3DenseVLTextForCausalLM), quantizada\n"
            "INT4 weight-only via torchao (Int4WeightOnlyConfig, group_size=128,\n"
            "packing_format=TILE_PACKED_TO_4D), 2026-07-21.\n\n"
            "So' os nn.Linear da torre de ENTENDIMENTO foram quantizados (exclui *_moe_gen*,\n"
            "que e' a torre de geracao/difusao/acao -- fica em bf16).\n\n"
            f"Tamanho serializado ANTES: {size_before/1e9:.3f} GB\n"
            f"Tamanho serializado DEPOIS: {size_after/1e9:.3f} GB (reducao {100*(1-size_after/size_before):.1f}%)\n\n"
            "Carregamento: NAO da' pra usar transformers.AutoModel padrao -- precisa\n"
            "reconstruir o Nemotron3DenseVLTextForCausalLM (via cosmos_framework) e\n"
            "chamar model.load_state_dict(torch.load('lm_int4_state_dict.pt'), strict=False)\n"
            "(os tensores viram AffineQuantizedTensor automaticamente ao serem carregados\n"
            "de volta -- torchao registra isso via torch.serialization).\n\n"
            "LIMITACAO: export para ONNX/TensorRT-Edge-LLM NAO FEITO -- o forward() desta\n"
            "classe usa uma interface customizada (SequencePack + metadados NATTEN + estado\n"
            "de memoria), incompativel com torch.onnx.export() direto. Precisaria de\n"
            "reverse-engineering adicional de _impl_generate_reasoner_text/_impl_reasoner_forward.\n"
        )
    print(f"[ok] salvo: {OUT_DIR}/lm_int4_state_dict.pt + README.txt", flush=True)

    return pipe


inf_mod.OmniInference.create = classmethod(_patched_create)

sys.argv = [
    "inference.py",
    "--parallelism-preset=latency",
    "-i", "inputs/workzone/gate_test.json", "inputs/workzone/ego_test.json",
    "inputs/workzone/active_test.json", "inputs/workzone/sign_test.json",
    "inputs/workzone/desc_test.json",
    "-o", "outputs/workzone_full_test_int4_torchao",
    "--checkpoint-path", "Cosmos3-Edge",
    "--seed=0",
    "--no-guardrails",
]
from cosmos_framework.scripts.inference import main  # noqa: E402

print("\n[run] rodando as 5 perguntas com o modelo INT4...", flush=True)
main()
print("[ok] CLI terminou normalmente", flush=True)

REF = {
    "gate_test": 'Got it, let\'s check the image. There\'s an orange "road work ahead" sign... So the answer should be yes.</think>\n\nYes',
    "ego_test": "...approaching...",
    "active_test": "...passive (sem trabalhadores visiveis)...",
    "sign_test": '...sign amarelo/vermelho (inconsistente entre perguntas na versao BF16 tambem)...',
    "desc_test": '...ROAD WORK IN PROGRESS sign, yellow warning sign, road surface changes...',
}

print("\n=== COMPARACAO INT4 vs BF16 (referencia, job 194667) ===")
for name in ["gate_test", "ego_test", "active_test", "sign_test", "desc_test"]:
    d = json.load(open(f"outputs/workzone_full_test_int4_torchao/{name}/sample_outputs.json"))
    answer = d["outputs"][0]["content"]["reasoner_text"]
    print(f"\n--- [{name}] ---")
    print(f"INT4:  {answer}")
    print(f"BF16 (referencia): {REF[name]}")

print("\n=== FIM ===")
