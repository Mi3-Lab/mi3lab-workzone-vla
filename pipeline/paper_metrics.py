"""Metrics matching arXiv:2606.08860 (Martinez-Sanchez, Ng, Maia et al.,
"Vision-Language Work Zone Intelligence...") Section V exactly, so any system
evaluated with this module reports numbers directly comparable to the paper's:

  - Frame-level accuracy + macro F1               (Sec V, "Frame-level accuracy
                                                     ... macro F1 score of 0.60")
  - Per-state IoU                                  (Sec V.A: "OUTSIDE IoU 0.56,
                                                     INSIDE IoU 0.47, APPROACHING
                                                     IoU 0.11, EXITING IoU 0.09")
  - Event-level Precision/Recall (Eq. 5-7)          ("INSIDE event recall of
                                                     96.5%, event precision 68.7%")
  - State entry timing offset Δt_s (Eq. 8)          (MAE 81f/2.7s, median 39f/1.3s
                                                     overall; INSIDE MAE 142f/4.7s)
  - Transition-tolerance matching (Table I)         (±{0,5,15,30} frame windows,
                                                     Precision/Recall/Accuracy)

All metrics are pooled across the whole evaluated video set (sum of counts,
not per-video-averaged), matching how the paper reports single dataset-wide
numbers over its 490-sequence evaluation set.
"""

from collections import namedtuple

Event = namedtuple("Event", ["start", "end"])


def extract_events(state_seq, target_state):
    """Contiguous runs of target_state in a per-frame state sequence."""
    events = []
    start = None
    for i, s in enumerate(state_seq):
        if s == target_state:
            if start is None:
                start = i
        elif start is not None:
            events.append(Event(start, i - 1))
            start = None
    if start is not None:
        events.append(Event(start, len(state_seq) - 1))
    return events


def _overlaps(a, b):
    return not (a.end < b.start or b.end < a.start)


def frame_accuracy(confusion, states):
    total = sum(confusion.values())
    correct = sum(confusion.get((s, s), 0) for s in states)
    return correct / total if total else 0.0


def macro_f1(confusion, states):
    """Unweighted mean of per-class F1 from a pooled (gt,pred)->count confusion dict."""
    f1s = []
    for s in states:
        tp = confusion.get((s, s), 0)
        fp = sum(v for (g, p), v in confusion.items() if p == s and g != s)
        fn = sum(v for (g, p), v in confusion.items() if g == s and p != s)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def per_state_iou(confusion, states):
    """IoU_s = TP_s / (TP_s + FP_s + FN_s), computed on frame-level presence of state s."""
    result = {}
    for s in states:
        tp = confusion.get((s, s), 0)
        fp = sum(v for (g, p), v in confusion.items() if p == s and g != s)
        fn = sum(v for (g, p), v in confusion.items() if g == s and p != s)
        result[s] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return result


class EventLevelAccumulator:
    """Pools TP_P/|P| and TP_G/|G| (Eq. 5-7) across many videos, per state."""

    def __init__(self, states):
        self.states = states
        self.tp_p = {s: 0 for s in states}
        self.n_p = {s: 0 for s in states}
        self.tp_g = {s: 0 for s in states}
        self.n_g = {s: 0 for s in states}

    def add_video(self, predicted_seq, gt_seq):
        for s in self.states:
            P = extract_events(predicted_seq, s)
            G = extract_events(gt_seq, s)
            self.n_p[s] += len(P)
            self.n_g[s] += len(G)
            self.tp_p[s] += sum(1 for p in P if any(_overlaps(p, g) for g in G))
            self.tp_g[s] += sum(1 for g in G if any(_overlaps(p, g) for p in P))

    def result(self):
        out = {}
        for s in self.states:
            precision = self.tp_p[s] / self.n_p[s] if self.n_p[s] else None
            recall = self.tp_g[s] / self.n_g[s] if self.n_g[s] else None
            out[s] = {"precision": precision, "recall": recall,
                      "n_pred_events": self.n_p[s], "n_gt_events": self.n_g[s]}
        return out


class TimingOffsetAccumulator:
    """Δt_s = t_pred_entry - t_gt_entry (Eq. 8) for matched (overlapping) event
    pairs -- each GT event matched to the overlapping predicted event whose
    start is closest to the GT event's start."""

    def __init__(self, states, fps=30.0):
        self.states = states
        self.fps = fps
        self.offsets = {s: [] for s in states}

    def add_video(self, predicted_seq, gt_seq):
        for s in self.states:
            P = extract_events(predicted_seq, s)
            G = extract_events(gt_seq, s)
            for g in G:
                if g.start == 0:
                    # The video simply *begins* in this state -- there is no
                    # preceding state to transition from, so this isn't a
                    # detection event at all. Without this guard, videos that
                    # start OUTSIDE (the overwhelming majority) trivially
                    # "match" at offset 0, deflating the OUTSIDE entry-timing
                    # error to something meaningless.
                    continue
                candidates = [p for p in P if _overlaps(p, g)]
                if not candidates:
                    continue
                best = min(candidates, key=lambda p: abs(p.start - g.start))
                self.offsets[s].append(best.start - g.start)

    def result(self):
        out = {}
        all_abs = []
        for s in self.states:
            vals = self.offsets[s]
            abs_vals = sorted(abs(v) for v in vals)
            all_abs.extend(abs_vals)
            if abs_vals:
                mae = sum(abs_vals) / len(abs_vals)
                median = abs_vals[len(abs_vals) // 2] if len(abs_vals) % 2 else \
                    (abs_vals[len(abs_vals) // 2 - 1] + abs_vals[len(abs_vals) // 2]) / 2
            else:
                mae = median = None
            out[s] = {"mae_frames": mae, "mae_s": mae / self.fps if mae is not None else None,
                      "median_frames": median,
                      "median_s": median / self.fps if median is not None else None,
                      "n": len(vals)}
        if all_abs:
            all_abs.sort()
            overall_mae = sum(all_abs) / len(all_abs)
            overall_median = all_abs[len(all_abs) // 2] if len(all_abs) % 2 else \
                (all_abs[len(all_abs) // 2 - 1] + all_abs[len(all_abs) // 2]) / 2
        else:
            overall_mae = overall_median = None
        out["overall"] = {"mae_frames": overall_mae,
                          "mae_s": overall_mae / self.fps if overall_mae is not None else None,
                          "median_frames": overall_median,
                          "median_s": overall_median / self.fps if overall_median is not None else None,
                          "n": len(all_abs)}
        return out


def _transition_points(state_seq):
    """Frame indices where state_seq[i] != state_seq[i-1] (i.e. i is the first
    frame of a new state)."""
    return [i for i in range(1, len(state_seq)) if state_seq[i] != state_seq[i - 1]]


class TransitionToleranceAccumulator:
    """Table I: greedy nearest-neighbor one-to-one matching of transition
    timestamps within +/-k frames. Accuracy = TP/(TP+FP+FN) (Jaccard-style),
    Precision = TP/|pred|, Recall = TP/|gt| -- the paper doesn't spell out the
    "Accuracy" formula explicitly; TP/(TP+FP+FN) is the standard IoU-style
    metric for point-matching problems and reproduces the reported ordering
    (Accuracy <= min(Precision, Recall) at every tolerance in their Table I)."""

    def __init__(self, tolerances=(0, 5, 15, 30)):
        self.tolerances = tolerances
        self.n_pred = 0
        self.n_gt = 0
        self.tp = {k: 0 for k in tolerances}

    def add_video(self, predicted_seq, gt_seq):
        pred_pts = _transition_points(predicted_seq)
        gt_pts = _transition_points(gt_seq)
        self.n_pred += len(pred_pts)
        self.n_gt += len(gt_pts)
        for k in self.tolerances:
            pairs = []
            for i, p in enumerate(pred_pts):
                for j, g in enumerate(gt_pts):
                    d = abs(p - g)
                    if d <= k:
                        pairs.append((d, i, j))
            pairs.sort()
            used_p, used_g = set(), set()
            for d, i, j in pairs:
                if i in used_p or j in used_g:
                    continue
                used_p.add(i)
                used_g.add(j)
                self.tp[k] += 1

    def result(self):
        out = {}
        for k in self.tolerances:
            tp = self.tp[k]
            fp = self.n_pred - tp
            fn = self.n_gt - tp
            precision = tp / self.n_pred if self.n_pred else 0.0
            recall = tp / self.n_gt if self.n_gt else 0.0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
            out[k] = {"precision": precision, "recall": recall, "accuracy": accuracy}
        return out


STATES = ["outside", "approaching", "inside", "exiting"]


def print_paper_report(confusion, event_acc, timing_acc, transition_acc, states=STATES):
    """Prints a report using the exact metric definitions of arXiv:2606.08860
    Section V, so the numbers are directly comparable to the paper's Table I /
    ablation numbers and headline INSIDE event recall/precision."""
    print("\n=== METRICAS NO FORMATO DO PAPER (arXiv:2606.08860) ===")

    print(f"\nAcuracia por-frame (todos os 4 estados): {frame_accuracy(confusion, states):.1%}")
    print(f"Macro F1: {macro_f1(confusion, states):.3f}")

    print("\n--- IoU por estado (Sec V.A) ---")
    for s, iou in per_state_iou(confusion, states).items():
        print(f"{s:14s} IoU={iou:.3f}")

    print("\n--- Precisao/Recall a nivel de EVENTO (Eq. 5-7) ---")
    for s, r in event_acc.result().items():
        p = f"{r['precision']:.1%}" if r["precision"] is not None else "N/A"
        rec = f"{r['recall']:.1%}" if r["recall"] is not None else "N/A"
        print(f"{s:14s} precision={p:>7s}  recall={rec:>7s}  "
              f"(n_pred_events={r['n_pred_events']}, n_gt_events={r['n_gt_events']})")

    print("\n--- Erro de timing de entrada Delta_t (Eq. 8, frames a 30fps) ---")
    for s, r in timing_acc.result().items():
        if r["mae_frames"] is not None:
            print(f"{s:14s} MAE={r['mae_frames']:.1f}f ({r['mae_s']:.2f}s)  "
                  f"mediana={r['median_frames']:.1f}f ({r['median_s']:.2f}s)  n={r['n']}")
        else:
            print(f"{s:14s} sem pares casados (n=0)")

    print("\n--- Matching de transicao por janela de tolerancia (Table I) ---")
    print(f"{'Tolerancia (f)':16s} {'Precision':>10s} {'Recall':>10s} {'Accuracy':>10s}")
    for k, r in transition_acc.result().items():
        print(f"{k:16d} {r['precision']:10.4f} {r['recall']:10.4f} {r['accuracy']:10.4f}")
