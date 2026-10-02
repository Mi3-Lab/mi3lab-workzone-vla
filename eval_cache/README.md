# eval_cache

Predicoes por frame, evidencia bruta e parametros ajustados. Cada `.npz` de
predicao tem `predicted` e `gt` por frame; os de evidencia tem `obs` com as
leituras cruas por ciclo.

Os scripts em `pipeline/` referenciam estes caminhos como `~/eval_cache/<nome>`,
que resolve por symlink para ca. Manter a estrutura plana por isso.

| diretorio | n | conteudo |
|---|---|---|
| `c3e_caldev` | 100 | C3E nos 100 videos de calibracao |
| `cosmos3edge` | 208 | C3E com limiares herdados do 2B (Fase 2, contaminado) |
| `cosmos3edge_cal` | 208 | C3E com limiares calibrados para ele (SISTEMA DO PAPER) |
| `evidence_2b` | 100 | evidencia bruta do 2B (tratamento simetrico) |
| `evidence_base` | 100 | evidencia bruta do checkpoint base (experimento pareado) |
| `evidence_c3e` | 100 | evidencia bruta por ciclo do C3E (calibracao da cascata) |
| `fusion_scores_cal` | 100 | scores da fusao YOLO+C3E (arquitetura descartada) |
| `gated_cheap` | 135 | variante de ciclo barato (DESCARTADA, incompleta) |
| `gated_hybrid` | 208 | hibrido com portao do detector (DESCARTADO: F1 0.427) |
| `joint_dump_cal` | 100 | evidencia do filtro conjunto, codificacao antiga (obsoleto) |
| `joint_dump_cal2` | 312 | evidencia do filtro conjunto, 312 videos de calibracao |
| `joint_dump_val` | 208 | evidencia do filtro conjunto, 208 de validacao |
| `joint_filter_val` | 208 | filtro conjunto, parametros antigos (obsoleto) |
| `joint_smoke` | 3 | smoke test de 3 videos |
| `joint_v2_val` | 208 | filtro conjunto, parametros de 100 videos de calibracao |
| `joint_full_val` | 208 | filtro conjunto, parametros de 312 videos (SISTEMA RECOMENDADO) |
| `ood_california` | 3 | logs do benchmark OOD, 39 sinais x 3 sistemas |
| `vlm` | 520 | nosso 2B, decodificacao amostrada (paper, Tabela 1 'raw') |
| `vlm_binarysign` | 208 | ablacao: SIGN binario em vez de transcricao |
| `vlm_greedy` | 208 | nosso 2B, greedy (paper, 'rec.') |
| `vlm_nofastentry` | 208 | ablacao: sem fast entry |
| `vlm_nofilter` | 208 | ablacao: sem filtro de qualificadores |
| `vlm_nosign` | 208 | ablacao: sem canal SIGN |
| `vlm_recal` | 208 | nosso 2B com limiares recalibrados (Fase 6) |
| `vlm_unrestbayes` | 208 | ablacao: filtro bayesiano irrestrito |
| `vlm_zeroshot` | 208 | checkpoint base sem fine-tuning (ablacao) |
| `yolo` | 44 | detector YOLO puro |
| `yolo_c3e_always` | 0 | C3E sempre ligado no lugar do CLIP (DESCARTADO) |
| `yoloclip` | 208 | detector+CLIP portado (paper, 'pair') |
| `yoloclip_dense` | 208 | detector em modo denso (decomposicao do gap do baseline) |
| `yoloclip_dev` | 100 | detector+CLIP nos 100 videos de calibracao |
| `yoloclip_fastentry` | 208 | detector + fast entry de texto (paper, '+FE') |
| `yoloclip_fastentry_dev` | 100 | detector+FE nos 100 de calibracao |

## Parametros e modelos

- `joint_params.json`
- `joint_params_full.json`
- `joint_params_v2.json`
- `stack_table.json`
- `tcn_full.pt`
- `tcn_lam0.5.pt`
- `tcn_lam1.5.pt`
- `tcn_lam4.0.pt`
- `tcn_model.pt`
- `tcn_model_v2.pt`
- `tcn_journal.pt` (treinado em joint_dump_cal2, avaliado nos 208 de validacao)
- `joint_params_nodet.json` (ablacao sem detector, 312 videos)
