"""Teste do modo REASONER do Cosmos3-Edge -- versao 2, corrigida.

Tentativa 1: usei `Cosmos3ForConditionalGeneration` do pacote leve
transformers-cosmos3 -- ele so registra model_type "cosmos3_omni", nao
"cosmos3_edge". Carregou uma arquitetura ERRADA silenciosamente (visual
encoder inteiro veio MISSING) -> saida em lixo.

Tentativa 2: troquei pra AutoModelForImageTextToText generico, mas sem
importar o modulo que registra "cosmos3_edge" -> AutoConfig nao reconhecia
o tipo, KeyError imediato.

Esta versao: importa cosmos_framework.model.generator.reasoner.cosmos3_edge
(codigo NATIVO do framework completo, commit de hoje 2026-07-20, "Cosmos3-
Edge + Distillation support") -- isso registra Cosmos3EdgeConfig +
Cosmos3EdgeForConditionalGeneration no Auto* corretamente, sem trust_remote_code.

Tentativa 3 (esta): a v2 anterior carregou a classe certa mas usou
AutoProcessor.from_pretrained() generico, que produziu image_grid_thw=None e
quebrou no forward. O proprio codigo do framework documenta (comentario em
cosmos3_edge_processing_test.py) que o processador NATIVO do transformers
corrompe silenciosamente as features de visao deste checkpoint -- e' preciso
usar build_cosmos3_edge_processor(), que reconstroi o processador exato a
partir dos arquivos do snapshot (tokenizer + chat_template.jinja + sub-configs),
nao o AutoProcessor generico.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import transformers
from huggingface_hub import snapshot_download

# importar isto registra "cosmos3_edge" no AutoConfig/AutoModelForImageTextToText
import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401
from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
    build_cosmos3_edge_processor,
)

from transformers import AutoModelForImageTextToText

IMAGE = "/data/wesleyferreiramaia/wokzone-alpamayo/models/jetson-deploy/test/sample_frame.jpg"

import os
_FULL_QUESTIONS = {
    "GATE": "Are there any road work indicators in this scene (cones, barriers, temporary signs, workers, work vehicles)? Answer Yes or No.",
    "EGO": "Is the ego vehicle currently outside, approaching, inside, or exiting a work zone?",
    "ACTIVE": "Is this an active work zone with workers present, or a passive zone?",
    "SIGN": "Read the text on any temporary traffic control signs visible in this scene.",
    "DESC": "Describe the work zone elements visible in this scene.",
}
# validacao rapida (1 pergunta) por padrao -- setar COSMOS3E_FULL=1 pra rodar as 5
QUESTIONS = _FULL_QUESTIONS if os.environ.get("COSMOS3E_FULL") == "1" else {"GATE": _FULL_QUESTIONS["GATE"]}

SEPARATOR = "-" * 20


def main():
    transformers.set_seed(0)

    print("[1] Baixando/localizando snapshot de nvidia/Cosmos3-Edge...", flush=True)
    snapshot_dir = snapshot_download("nvidia/Cosmos3-Edge")
    print(f"    snapshot em: {snapshot_dir}", flush=True)

    print("[2] Carregando modelo (classe nativa do framework)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot_dir, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
    print(f"    classe carregada: {type(model).__name__}", flush=True)

    print("[3] Construindo o processador NATIVO (build_cosmos3_edge_processor, "
          "NAO o AutoProcessor generico -- este ultimo corrompe as features de "
          "visao deste checkpoint, ver cosmos3_edge_processing_test.py)...", flush=True)
    processor = build_cosmos3_edge_processor(snapshot_dir)
    print("    OK\n", flush=True)

    from PIL import Image
    image = Image.open(IMAGE).convert("RGB")

    for tag, question in QUESTIONS.items():
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        generated = model.generate(**inputs, max_new_tokens=256)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated, strict=False)]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]

        print(SEPARATOR)
        print(f"[{tag}] {question}")
        print(f"  -> {answer}")

    print(SEPARATOR)
    print("\nreferencia (nosso 2B stage7_1, mesma cena Boston): 4/4 zonas detectadas, "
          "DESC descreve elementos corretamente, 0% degeneracao")


if __name__ == "__main__":
    main()
