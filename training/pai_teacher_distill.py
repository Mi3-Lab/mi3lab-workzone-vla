"""PAIDataset com targets de destilacao do professor 10B.

Substitui ego_future_xyz/rot pela trajetoria do professor (gerada por
gen_teacher_traj_labels.py) quando o professor ficou perto o bastante do GT
(teacher_ade_to_gt <= max_teacher_ade); caso contrario mantem o GT.

E' o "target mixing" da receita Phase-1 documentada pelo time que deployou
o Alpamayo no Thor: teacher como alvo polido, GT como prioridade quando o
professor diverge. A troca acontece ANTES da tokenizacao (o preprocess gera
os tokens discretos de traj_future a partir do ego_future ja substituido),
entao tanto a loss do LM (tokens) quanto a do flow-matching (acao continua)
veem o mesmo alvo.
"""
import glob
import os

import torch
from alpamayo.data.pai import PAIDataset


class PAIDatasetTeacherDistill(PAIDataset):
    def __init__(
        self,
        labels_dir: str,
        max_teacher_ade: float = 1.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_teacher_ade = max_teacher_ade
        self.labels = {}
        for p in sorted(glob.glob(os.path.join(labels_dir, "shard_*.pt"))):
            self.labels.update(torch.load(p))
        n_ok = sum(1 for v in self.labels.values()
                   if v["teacher_ade_to_gt"] <= max_teacher_ade)
        print(f"[PAIDatasetTeacherDistill] {len(self.labels)} labels carregados de "
              f"{labels_dir}; {n_ok} passam no filtro ade<={max_teacher_ade}m "
              f"(o resto usa GT)")

    def __getitem__(self, idx):
        # adia o preprocess pra DEPOIS da substituicao do alvo
        fn = self.vla_preprocess_func
        self.vla_preprocess_func = None
        try:
            sample = super().__getitem__(idx)
        finally:
            self.vla_preprocess_func = fn
        if sample is None:
            return None

        lbl = self.labels.get(sample["clip_id"])
        if lbl is not None and lbl["teacher_ade_to_gt"] <= self.max_teacher_ade:
            # shapes pos-squeeze do PAIDataset: ego_future_xyz [G,64,3]
            sample["ego_future_xyz"] = lbl["teacher_xyz"].unsqueeze(0).to(
                sample["ego_future_xyz"].dtype)
            sample["ego_future_rot"] = lbl["teacher_rot"].unsqueeze(0).to(
                sample["ego_future_rot"].dtype)

        if fn is not None:
            sample["tokenized_data"] = fn(data=sample)
        return sample
