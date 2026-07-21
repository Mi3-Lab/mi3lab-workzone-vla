#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: OpenMDW-1.1
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate==1.13.0",
#   "torch==2.11.0",
#   "torchvision",
#   "transformers==5.8.0",
#   "transformers-cosmos3",
# ]
# [tool.uv.sources]
# transformers-cosmos3 = { path = "/data/wesleyferreiramaia/wokzone-alpamayo/cosmos-framework/packages/transformers-cosmos3", editable = true }
# ///

"""Teste do modo REASONER do Cosmos3-Edge nas mesmas 5 perguntas v16 que
treinamos no nosso 2B (workzone-2b-stage7-1), pra comparar qualidade de
deteccao/raciocinio de workzone SEM nenhum fine-tuning -- e' o Cosmos3-Edge
"de fabrica" contra nosso modelo especializado.

Usa a mesma imagem real de teste do pacote de deploy (Boston, dentro da
zona de obra, ja validada em outros testes deste projeto).
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
import transformers_cosmos3  # noqa: F401  (registra cosmos3_edge no Auto*)

IMAGE = "/data/wesleyferreiramaia/wokzone-alpamayo/models/jetson-deploy/test/sample_frame.jpg"

# as mesmas 5 categorias v16 que treinamos no nosso 2B (ver
# RELATORIO_MODELO_2B_WORKZONE.md secao 3.3), pra comparacao direta
QUESTIONS = {
    "GATE": "Are there any road work indicators in this scene (cones, barriers, temporary signs, workers, work vehicles)? Answer Yes or No.",
    "EGO": "Is the ego vehicle currently outside, approaching, inside, or exiting a work zone?",
    "ACTIVE": "Is this an active work zone with workers present, or a passive zone?",
    "SIGN": "Read the text on any temporary traffic control signs visible in this scene.",
    "DESC": "Describe the work zone elements visible in this scene.",
}

SEPARATOR = "-" * 20


def main():
    transformers.set_seed(0)

    print("[1] Carregando nvidia/Cosmos3-Edge (trust_remote_code, classe via Auto*)...", flush=True)
    model_name = "nvidia/Cosmos3-Edge"
    # AutoModelForImageTextToText resolve a classe certa (Cosmos3EdgeForConditionalGeneration)
    # via config.architectures, em vez de hardcodar (bug da vez anterior: usar
    # Cosmos3ForConditionalGeneration, classe do Nano, causava visual encoder
    # inteiro MISSING -> saida em lixo).
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    print(f"    classe carregada: {type(model).__name__}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("    OK\n", flush=True)

    for tag, question in QUESTIONS.items():
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE},
                {"type": "text", "text": question},
            ],
        }]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"fps": 4},
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
