#!/usr/bin/env python3
"""Draft labels, written from contact sheets by the draft annotator (Claude),
following GUIDELINE.md.  Times are seconds on a 2 s grid; each segment is
[start, end) in seconds.  Converted to the benchmark's frame-interval format by
build_draft.py.  NOT ground truth until verified with label_tool.py.
"""

DRAFT = {
    "Merced_Day_01": {
        "states": [
            ("outside", 0, 138),
            ("approaching", 138, 188),
            ("inside", 188, 192),
            ("exiting", 192, 194),
            ("approaching", 194, 208),
            ("inside", 208, 336),
            ("exiting", 336, 340),
            ("outside", 340, 361),
        ],
        "visible": [(0, 40), (46, 68), (72, 76), (130, 132), (138, 340)],
        "notes": [
            (0, 68, "cones, roller and tubular markers on the OPPOSITE carriageway; "
                    "human sign timestamps at 22 s and 55 s fall here, no ego-facing "
                    "sign readable at full resolution -> outside"),
            (144, 188, "stopped at a red light after ROAD WORK AHEAD; devices ahead "
                       "hard to see at sheet resolution"),
            (188, 194, "barricades and message board on the ego's right shoulder, "
                       "judged ego-relevant"),
        ],
    },
    "CA99_Night_01": {
        "states": [
            ("outside", 0, 44),
            ("approaching", 44, 130),
            ("inside", 130, 150),
            ("exiting", 150, 154),
            ("outside", 154, 181),
        ],
        "visible": [(44, 48), (64, 66), (80, 86), (94, 100), (126, 138), (144, 150)],
        "notes": [
            (0, 44, "human sign timestamp at 0 s; no work-zone sign readable in the "
                    "first frames; tiny orange shapes at 22 s and 38 s left as no"),
            (44, 130, "LOW CONFIDENCE: freeway at night, only orange diamond signs on "
                      "the shoulder are visible; no devices; kept as one approach"),
            (130, 150, "LOW CONFIDENCE: unlit drums/channelizers on the right shoulder "
                       "and signs under the overpass; judged inside"),
        ],
    },
    "CA50_Day_01": {
        "states": [
            ("outside", 0, 20),
            ("approaching", 20, 22),
            ("inside", 22, 28),
            ("exiting", 28, 30),
            ("outside", 30, 84),
            ("approaching", 84, 144),
            ("inside", 144, 181),
        ],
        "visible": [(20, 28), (74, 78), (84, 181)],
        "notes": [
            (20, 28, "cones close the LEFT lanes of the ego's direction, message "
                     "board on a work truck; ego-relevant (this is the teaser frame)"),
            (74, 78, "drums and an orange sign near an on-ramp; judged not ego-relevant"),
            (126, 144, "between the orange diamond signs and the first drums, no "
                       "devices visible; kept as approaching"),
        ],
    },
    "CA50_Day_02": {
        "states": [
            ("outside", 0, 8),
            ("approaching", 8, 10),
            ("inside", 10, 18),
            ("exiting", 18, 20),
            ("outside", 20, 60),
            ("approaching", 60, 66),
            ("inside", 66, 122),
            ("exiting", 122, 126),
            ("outside", 126, 160),
        ],
        "visible": [(8, 18), (34, 40), (60, 122)],
        "notes": [
            (10, 18, "drums along the left edge of the ego's direction; judged inside"),
            (34, 40, "drums on the far left of a wide freeway; judged not ego-relevant"),
            (66, 122, "long zone: orange signs, drums on both shoulders, left-lane "
                      "closure at 96-100 s; some gaps with no devices kept as inside"),
        ],
    },
    "CA99_Day_01": {
        "states": [
            ("outside", 0, 12),
            ("approaching", 12, 58),
            ("outside", 58, 78),
        ],
        "visible": [(12, 14), (22, 34), (40, 44), (54, 56), (60, 62)],
        "notes": [
            (0, 12, "strong sun glare"),
            (12, 58, "LOW CONFIDENCE: orange diamond signs and sign stands on the "
                     "right shoulder repeatedly, but no devices in the lanes; no zone "
                     "is ever entered in this clip, so approaching throughout"),
            (60, 62, "excavator beyond the barrier; judged not ego-relevant"),
        ],
    },
    "CA99_Day_02": {
        "states": [
            ("outside", 0, 114),
            ("approaching", 114, 160),
            ("inside", 160, 174),
            ("exiting", 174, 176),
            ("outside", 176, 181),
        ],
        "visible": [(6, 8), (62, 64), (114, 116), (124, 174)],
        "notes": [
            (0, 12, "human sign timestamp at 4 s; drums on the right shoulder at 6 s "
                    "only, no readable ego-facing sign; left as outside"),
            (62, 64, "orange diamond signs far away on the right; not counted"),
            (114, 160, "orange diamond signs on both sides, SPEED 55 at 124 s, bridge "
                       "construction with a crane ahead; no devices in lanes yet"),
            (160, 174, "temporary concrete barriers, bridge falsework and utility "
                       "trucks beside the ego's lanes"),
        ],
    },
    "CA99_Evening_01": {
        "states": [
            ("outside", 0, 24),
            ("approaching", 24, 86),
            ("outside", 86, 108),
        ],
        "visible": [(24, 28), (48, 52), (56, 58), (70, 72)],
        "notes": [
            (24, 86, "LOW CONFIDENCE: dusk; orange diamond signs on both shoulders at "
                     "24-26 s and again far ahead at 48-72 s, no devices seen at sheet "
                     "resolution; human sign timestamps at 66 and 77 s"),
        ],
    },
    "Merced_Day_02": {
        "states": [
            ("outside", 0, 12),
            ("approaching", 12, 16),
            ("outside", 16, 86),
            ("approaching", 86, 138),
            ("inside", 138, 181),
        ],
        "visible": [(12, 16), (86, 90), (110, 114), (120, 122), (138, 181)],
        "notes": [
            (12, 16, "LOW CONFIDENCE: portable message board (\"...DELAY\") and a cone "
                     "on the right curb; then a long stop at an intersection with no "
                     "devices visible, labeled outside"),
            (86, 138, "ROAD WORK AHEAD sign at 88 s; human sign timestamps at 104 and "
                      "124 s; devices only at the far-right corners until 138 s"),
            (138, 181, "cones and channelizers in the lanes, RIGHT LANE ENDS, "
                       "message board WORK ZONE; lane closure through the end"),
        ],
    },
    "CA99_Night_02": {
        "states": [
            ("outside", 0, 2),
            ("approaching", 2, 36),
            ("inside", 36, 80),
            ("exiting", 80, 84),
            ("outside", 84, 181),
        ],
        "visible": [(2, 4), (8, 16), (36, 80), (138, 140), (156, 166)],
        "notes": [
            (2, 36, "orange diamond signs on the right shoulder at 2-14 s; human sign "
                    "timestamps at 13 and 15 s"),
            (36, 80, "LOW CONFIDENCE: drums along the left edge (median shoulder) of "
                     "the ego's direction, signs on the right; judged inside"),
            (138, 166, "orange signs on the exit ramp and surface streets after "
                       "leaving the freeway; judged not ego-relevant"),
        ],
    },
    "CA99_Night_03": {
        "states": [
            ("outside", 0, 16),
            ("approaching", 16, 36),
            ("outside", 36, 62),
        ],
        "visible": [(16, 28)],
        "notes": [
            (0, 16, "human sign timestamps at 4 and 13 s; headlight glare, no work "
                    "zone sign readable at sheet resolution"),
            (16, 36, "LOW CONFIDENCE: orange diamond signs on both shoulders at "
                     "16-26 s (human timestamps 24 and 30 s); no devices seen; "
                     "possibly signs for the zone of CA99_Night_02"),
        ],
    },
    "CA99_Night_04": {
        "states": [
            ("outside", 0, 181),
        ],
        "visible": [(108, 110), (128, 130)],
        "notes": [
            (0, 20, "LOW CONFIDENCE: human sign timestamp at 10 s marked Hard; no "
                    "work-zone sign readable at sheet resolution; check at full size"),
            (108, 130, "amber flashing light on the right shoulder and a row of "
                       "reflectors; could be a work vehicle, not ego-relevant as seen"),
        ],
    },
    "Fresno_Sunset_01": {
        "states": [
            ("outside", 0, 24),
            ("approaching", 24, 44),
            ("inside", 44, 94),
            ("exiting", 94, 96),
            ("outside", 96, 194),
            ("approaching", 194, 220),
            ("inside", 220, 276),
            ("exiting", 276, 278),
            ("outside", 278, 316),
        ],
        "visible": [(16, 18), (24, 26), (28, 94), (194, 196), (206, 276)],
        "notes": [
            (24, 44, "orange fencing and barricades ahead at an intersection; "
                     "human sign timestamps at 50 and 51 s"),
            (44, 94, "barricades, bridge construction, temporary concrete barriers "
                     "along the lanes, channelizers in the lanes at 84-92 s; "
                     "LOW CONFIDENCE at 72-84 s (only concrete barriers)"),
            (98, 156, "long stop at a red light, no work visible"),
            (194, 220, "orange diamond signs on the right; human timestamp 211 s"),
            (220, 276, "barricades, K-rail, cranes; strong sun glare 236-270 s; "
                       "human timestamp 239 s"),
        ],
    },
    "Merced_NightFog_01": {
        "states": [
            ("outside", 0, 36),
            ("approaching", 36, 72),
            ("inside", 72, 100),
            ("exiting", 100, 102),
            ("outside", 102, 138),
            ("approaching", 138, 144),
            ("outside", 144, 246),
            ("approaching", 246, 258),
            ("inside", 258, 361),
        ],
        "visible": [(36, 40), (60, 64), (72, 100), (138, 142), (166, 168),
                    (192, 202), (240, 242), (256, 361)],
        "notes": [
            (36, 72, "orange diamond sign on the right at 36-38 s, barricade at the "
                     "stop at 60-64 s"),
            (72, 100, "LOW CONFIDENCE: cones and equipment along the LEFT edge "
                      "(median side) of the ego's road; judged inside"),
            (104, 112, "human sign timestamp at 108 s; nothing readable in fog"),
            (138, 144, "LOW CONFIDENCE: SPEED 25 sign and cones on the left at "
                       "140 s (human timestamp 140 s)"),
            (192, 202, "cones in a parking lot; not ego-relevant"),
            (246, 361, "arrow board at 258 s, cones and channelizers along the "
                       "right edge, WORK ZONE 25 MPH message board at 300-306 s; "
                       "human timestamps 229, 247 and 280 s"),
        ],
    },
    "Merced_NightRain_01": {
        "states": [
            ("outside", 0, 160),
            ("approaching", 160, 230),
            ("outside", 230, 272),
            ("approaching", 272, 310),
            ("inside", 310, 340),
            ("exiting", 340, 344),
            ("outside", 344, 358),
            ("approaching", 358, 366),
            ("inside", 366, 424),
            ("exiting", 424, 428),
            ("outside", 428, 541),
        ],
        "visible": [(78, 92), (120, 150), (162, 166), (198, 202), (272, 276),
                    (288, 292), (310, 340), (358, 424)],
        "notes": [
            (78, 92, "orange diamond signs on the right, no zone follows; kept outside"),
            (120, 150, "cones and a work truck left of the double yellow line "
                       "(opposite direction); not ego-relevant"),
            (160, 230, "LOW CONFIDENCE: SPEED 25 signs at 164 and 200 s (human "
                       "timestamps 163 and 200 s), a few cones at 200 s"),
            (272, 310, "ROAD WORK AHEAD at 274 s, barricade at 290 s; human 298 s"),
            (310, 340, "cones and channelizers in and beside the ego's lanes; "
                       "human 332 s"),
            (358, 424, "STOP message board, cones along the right edge, orange "
                       "diamond sign, work truck"),
        ],
    },
}
