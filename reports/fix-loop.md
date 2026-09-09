# Fix loop

## 1. Worst numbered gate

**Photo-tier wall length, `room_01`, 4.0 m tape axis.**

| | |
| --- | --- |
| Tape | 4.00 m × 3.40 m |
| Before | **3.48 m × 3.25 m** (`reports/fix-loop/before_room_01.json`) |
| Error on 4.00 m axis | **−13.0%** (outside photo ±8%) |

Hallway walls were already inside ±8%. This was the worst numbered wall fail.

## 2. Hypothesis and evidence

The floor polygon is a min-area rectangle of a near-floor point slice, then clipped to a radius from the median. At the **80th percentile**, furniture (wardrobe/bed) is the dense core; plaster walls are sparse and get dropped. Evidence: the short 3.48 m side vs 4.00 m tape; Polycam’s same bedroom also disagrees with tape on the other axis (furniture hulls).

A first attempt that **also** mixed a wall-height band overshot to **5.90 m**. That band was **reverted**. The shipped fix is the clip only.

## 3. Fix shipped

`cozmo/reconstruct/planes.py`: radius clip **80 → 96**.

## 4. Prediction vs after

Predicted: 4.00 m axis inside ±8% (3.68–4.32 m). Ceiling **not** 1.5 cm.

| Axis | Tape | Before | After (`out/fix-after`) | After err | ±8% |
| --- | --- | --- | --- | --- | --- |
| long | 4.00 | 3.48 | **4.07** | +1.7% | **pass** |
| short | 3.40 | 3.25 | **3.62** | +6.4% | **pass** |

Ceiling still **4.10 m** vs 2.45 m tape — fail, as predicted. After JSON: `reports/fix-loop/after_room_01.json`.

## 5. Regenerate

```bash
source .venv312/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m cozmo run captures/ --out out/fix-after --only room_01 --backend vggt
```

Before snapshot is the pre-fix `out/plan.json` extract. After is the command above.
