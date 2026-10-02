#!/usr/bin/env python3
"""Prepara um grafo de chunk do VAE para o builder do TensorRT.

Faz duas coisas, ambas via constant folding do ONNX Runtime (otimização de
grafo -- não executa convolução nenhuma, é rápido e não precisa de GPU):

  1. **Remove o nó `If`** de `mid_block/attentions`, cujos ramos produzem
     tensores de RANK diferente. O parser do TensorRT rejeita isso
     (`IIfConditionalOutputLayer inputs must have the same shape`). Como o
     grafo do chunk é totalmente estático, a condição do `If` é constante e
     o folding elimina o nó.

     ATENÇÃO: NÃO tente "consertar" igualando o rank dos ramos com um
     `Unsqueeze`. Isso satisfaz o parser mas muda o rank de um tensor e
     quebra um `Transpose` mais adiante -- o grafo passa a buildar e rodar
     no TensorRT produzindo resultado errado em silêncio (o ONNX Runtime
     recusa executá-lo, foi assim que o defeito apareceu).

  2. **Dobra os `Pad` no atributo `pads` do `Conv` seguinte**, eliminando os
     nós `Pad`. Seguro porque todo `Pad` do VAE é `mode="constant"` com
     valor `0.0`, então `Pad(0) + Conv(pads=0)` ≡ `Conv(pads=P)`.

Efeito típico: 1834 -> 533 nós, `If` 1 -> 0, `Pad` 34 -> 4.

Uso: python fold_vae_chunk_graph.py <entrada.onnx> <saida.onnx>
"""
from __future__ import annotations

import os
import sys
import tempfile

import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


def count(graph):
    from collections import Counter
    c = Counter(n.op_type for n in graph.node)
    return len(graph.node), c.get("Conv", 0), c.get("Pad", 0), c.get("If", 0)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]

    m = onnx.load(in_path)
    n, cv, pd, iff = count(m.graph)
    print(f"    entrada: {n} nós, Conv={cv}, Pad={pd}, If={iff}", flush=True)

    # value_info antigo carrega shapes obsoletas que atrapalham a reinferência
    del m.graph.value_info[:]
    for out in m.graph.output:
        out.type.tensor_type.ClearField("shape")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_in = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_out = f.name
    onnx.save(m, tmp_in, save_as_external_data=False)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath = tmp_out
    _s = ort.InferenceSession(tmp_in, sess_options=so, providers=["CPUExecutionProvider"])
    del _s

    m2 = onnx.load(tmp_out)
    g = m2.graph

    by_output = {o: nd for nd in g.node for o in nd.output}
    inits = {i.name: i for i in g.initializer}

    def const_of(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        src = by_output.get(name)
        if src is not None and src.op_type == "Constant":
            return numpy_helper.to_array(src.attribute[0].t)
        return None

    folded = 0
    to_remove = []
    for nd in g.node:
        if nd.op_type != "Pad" or len(nd.input) < 2:
            continue
        pads = const_of(nd.input[1])
        if pads is None:
            continue
        consumers = [c for c in g.node if nd.output and nd.output[0] in c.input]
        if len(consumers) != 1 or consumers[0].op_type != "Conv":
            continue
        conv = consumers[0]
        if conv.input[0] != nd.output[0]:
            continue

        pads = pads.tolist()
        rank = len(pads) // 2
        begin, end = pads[:rank], pads[rank:]
        ks = pads_attr = None
        for a in conv.attribute:
            if a.name == "kernel_shape":
                ks = list(a.ints)
            if a.name == "pads":
                pads_attr = a
        if ks is None:
            continue
        lead = rank - len(ks)
        if any(v for v in begin[:lead]) or any(v for v in end[:lead]):
            continue  # padding fora das dims espaciais: não dobra
        if pads_attr is not None and any(v != 0 for v in pads_attr.ints):
            continue  # Conv já tem padding próprio: somar mudaria a semântica

        conv_pads = begin[lead:] + end[lead:]
        if pads_attr is not None:
            del pads_attr.ints[:]
            pads_attr.ints.extend(conv_pads)
        else:
            conv.attribute.append(helper.make_attribute("pads", conv_pads))
        conv.input[0] = nd.input[0]
        to_remove.append(nd)
        folded += 1

    for nd in to_remove:
        g.node.remove(nd)

    n, cv, pd, iff = count(g)
    print(f"    saída:   {n} nós, Conv={cv}, Pad={pd}, If={iff}  ({folded} Pad dobrados)", flush=True)
    if iff:
        print(f"    [aviso] ainda restam {iff} nó(s) If -- o build do TensorRT pode falhar", flush=True)

    onnx.save(m2, out_path, save_as_external_data=False)
    for t in (tmp_in, tmp_out):
        try:
            os.unlink(t)
        except OSError:
            pass


if __name__ == "__main__":
    main()
