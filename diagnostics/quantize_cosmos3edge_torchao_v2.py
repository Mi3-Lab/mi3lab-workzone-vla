"""v2: quantiza o Cosmos3-Edge com torchao (arquitetura nativa, sem converter
pra Qwen3VL) e deixa o CLI OFICIAL rodar a geracao normalmente (mesmo
caminho ja' provado funcionar o dia todo) -- em vez de tentar chamar
.generate() na mao (Nemotron3DenseVLTextForCausalLM nao tem essa interface
padrao).

Estrategia: intercepta OmniInference.create, quantiza o model.net.language_model
IN-PLACE ali dentro, e deixa create() retornar o pipe NORMALMENTE (sem
SystemExit) -- dai' o resto do main() do CLI segue seu fluxo normal e gera a
resposta de verdade com o modelo ja' quantizado.

Compara contra a resposta BOA que ja' temos de hoje pra mesma pergunta GATE
(job 194663, checkpoint nao quantizado): "Yes" com raciocinio mencionando o
sign "road work ahead".
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch

import cosmos_framework.inference.inference as inf_mod

_orig_create = inf_mod.OmniInference.create.__func__


def real_serialized_bytes(m):
    """Mede o tamanho REAL via torch.save -- .numel()*.element_size() mente
    pra tensor subclasses do torchao (AffineQuantizedTensor/Int4Tensor
    reportam o dtype/numel LOGICO original pra compatibilidade de API, nao
    o armazenamento real comprimido)."""
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
    print(f"[quant] tamanho ANTES (serializado real, language_model): {size_before / 1e9:.3f} GB", flush=True)

    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
    # Int4PackingFormat.PLAIN (default) usa Int4Tensor, que exige o pacote
    # interno 'mslk' da Meta (nao disponivel via pip real). TILE_PACKED_TO_4D
    # usa Int4TilePackedTo4dTensor, que NAO depende de mslk.
    qconfig = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)

    def _filter_fn(module, fqn):
        if not isinstance(module, torch.nn.Linear):
            return False
        if "moe_gen" in fqn:
            return False
        return True

    n_eligible = sum(1 for name, mod in lm.named_modules() if _filter_fn(mod, name))
    print(f"[quant] {n_eligible} camadas nn.Linear elegiveis (fora _moe_gen)", flush=True)

    quantize_(lm, qconfig, filter_fn=_filter_fn)
    print("[quant] quantize_() concluido", flush=True)

    size_after = real_serialized_bytes(lm)
    print(f"[quant] tamanho DEPOIS (serializado real): {size_after / 1e9:.3f} GB "
         f"(reducao: {100 * (1 - size_after / size_before):.1f}%)", flush=True)

    return pipe


inf_mod.OmniInference.create = classmethod(_patched_create)

sys.argv = [
    "inference.py",
    "--parallelism-preset=latency",
    "-i", "inputs/workzone/gate_test.json",
    "-o", "outputs/workzone_gate_test_torchao_int4",
    "--checkpoint-path", "Cosmos3-Edge",
    "--seed=0",
    "--no-guardrails",
]
from cosmos_framework.scripts.inference import main  # noqa: E402

print("[run] rodando o CLI oficial com o modelo quantizado in-place...", flush=True)
main()
print("[ok] CLI terminou normalmente", flush=True)

import json
d = json.load(open("outputs/workzone_gate_test_torchao_int4/gate_test/sample_outputs.json"))
answer = d["outputs"][0]["content"]["reasoner_text"]
print(f"\n=== RESPOSTA (modelo quantizado INT4) ===")
print(answer)
print(f"\n=== REFERENCIA (job 194663, modelo NAO quantizado, mesma pergunta) ===")
print('Got it, let\'s check the image. There\'s an orange "road work ahead" sign, '
     'which is a temporary sign. ... So the answer should be yes.</think>\n\nYes')
