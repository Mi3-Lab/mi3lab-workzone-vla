# California work-zone labels: annotation guideline

Two labels per time step, from the dashcam image alone. The annotator does not look at any model output.

## Label A: work zone visible (yes / no)

Yes if any temporary traffic-control element or active work is visible anywhere in the image: cones, drums, tubular markers, barricades, temporary barriers, temporary signs (orange diamond signs, work-zone speed limits, arrow boards, message boards), workers, or construction equipment at work. Permanent infrastructure (a permanent concrete median, a regular speed-limit sign, a parked truck with no devices around it) is **no**.

## Label B: ego-relevant (yes / no)

Yes if the work zone regulates or physically affects the ego vehicle's direction of travel:

- a lane of the ego's direction is closed, shifted or narrowed by devices;
- devices, workers or equipment are in or directly beside the ego's lanes (shoulder of the ego's direction included);
- a work-zone sign addressed to the ego's direction is visible or has just been passed (ROAD WORK AHEAD, work-zone speed limit, LANE CLOSED, flagger ahead).

No if the work is only on the opposite carriageway, behind a permanent barrier, on the sidewalk, or on a side street the ego does not enter.

## Four states (from label B, same convention as the Boston/Seattle benchmark)

- **approaching**: an ego-relevant work zone is signed or visible ahead, but the ego is not yet beside the first device.
- **inside**: the ego is beside ego-relevant devices, workers or equipment, or between them.
- **exiting**: the ego has passed the last ego-relevant device and the zone is receding (END ROAD WORK, or devices only behind or at the edge of view), for at most a few seconds.
- **outside**: everything else.

## Procedure

1. Draft pass: contact sheets at one frame every 2 s (`contact_sheets.py`). Human sign timestamps from `Speed_Sign_Timestamps.txt` may be used as anchors because they are human labels. Draft file: `california_draft.json`, marked `"verified": false`.
2. Verification: a person steps through the video with `label_tool.py`, accepts or moves each transition, and fixes label A where needed. Verified file: `california_labels.json`.
3. Both files record who produced them. Only verified labels are used for reported results.

## Known hard cases

- Long urban arterials with cones along the opposite side: A = yes, B = no, unless a sign addressed to the ego posts a work-zone speed limit.
- Night and fog: if devices cannot be seen, label what is visible; do not infer from later frames.
- Small signs at 640 px may be unreadable in contact sheets; check them at full resolution in the tool.
