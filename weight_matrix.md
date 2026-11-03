# Weight Matrix

Accuracy of the app, and of different LBP, HOG, and grid-box weights, on the `D:\face_test` photos. Tested on 2026-11-03 on temporary copies.

**Applied on 2026-11-03: LBP 0.4 / HOG 0.6.** Sections 1–5 record the test that led to this change, when the app used LBP 0.6 / HOG 0.4. Section 7 has the results after the change.

## Current approach

1. **Find the face:** YuNet finds the face and 5 landmarks. A photo with another face at 0.8 confidence or higher is rejected.
2. **Check quality:** the app checks face size, brightness, sharpness, shadow, and head position.
3. **Prepare the face:** the landmarks align the face to a 128×128 grayscale image, and CLAHE evens out the lighting.
4. **Create numbers:** the face is split into 64 boxes, giving 640 LBP numbers for texture and 512 HOG numbers for edges and shape.
5. **Compare:** each box gets a chi-square distance for LBP and a cosine distance for HOG. The closest share of boxes is averaged for each.
6. **Combine:** `Final distance = LBP weight × LBP distance + HOG weight × HOG distance`.
7. **Decide:** a person is verified only if their distance is within the limit, and at least the margin below the second-closest person.

### Current weights and settings

| Setting | Value | Meaning |
|---|---:|---|
| LBP weight | 0.4 | Share of the final distance from texture |
| HOG weight | 0.6 | Share of the final distance from edges and shape |
| Boxes kept | 0.8 (51 of 64) | Closest boxes averaged, with the worst 13 ignored |
| `MATCH_DISTANCE_CUTOFF` | 0.055 | Maximum distance for a match |
| `MATCH_CLEAR_WINNER_MARGIN` | 0.004 | Minimum gap to the second-closest person |
| `MATCH_PERSONAL_LIMIT_FACTOR` | 1.1 | Personal limit = 1.1 × distance from the closest person's registered photo to the nearest other registered photo |

## Test setup

- **Registered people (7):** Arpit Patel, Jony Deep, Krish Hemsworth, Mit Patel, rahul Gandhi, RDJ, and Tom Holland, with one registered photo each.
- **Test photos (20):** photos 1, 2, and 3 from each folder. `rahul Gandhi\1.jpg` is left out because it's the same file as his registered photo.
- **Photo checks:** 17 of 20 passed. `krish Hemsworth\1.jpg`, `rahul Gandhi\3.jpg`, and `tom\3.jpg` were rejected for shadow.
- **Registered-person test (17 scans):** each accepted photo is scanned with all 7 people registered.
- **Stranger test (17 scans):** each accepted photo is scanned with that person removed, so any verification is an error.
- **Held-out test:** settings are chosen without one person, then tested on that person, and this repeats for each person. It's the fairest result here, because settings aren't chosen using the photos they're tested on.

| Term | Meaning | Better when |
|---|---|---|
| **AUC** | Chance that a same-person comparison is closer than a different-person comparison. 0.5 is guessing | Higher |
| **EER** | Error rate where wrong accepts and wrong rejects are equal | Lower |
| **Closest** | The closest registered person is the right person, with no limit applied | Higher |
| **Correct** | Verified as the right person | Higher |
| **Wrong** | Verified as a different registered person | Lower |
| **Strangers** | Verified after their own registration is removed | Lower |

---

## 1. Baseline results (LBP 0.6 / HOG 0.4, before the change)

The real API on a temporary copy gave the same decisions as the test script for all 20 photos.

| Result | Count |
|---|---:|
| **Correct** | **5 of 20 (25%)** |
| **Wrong person verified** | **1 of 20 (5%)** |
| Please retake | 4 |
| Unknown person | 7 |
| Photo rejected | 3 |
| **Strangers verified** | **4 of 17 (24%)** |
| Closest correct | 8 of 17 (47%) |
| AUC | 0.729 |
| EER | 41.2% |
| Latency, 62 scans | Median 46 ms, fastest 22 ms, slowest 448 ms (the first scan) |
| RAM, normal / peak | 102 / 113 MB |

### Per person

| Person | Photo 1 | Photo 2 | Photo 3 |
|---|---|---|---|
| Mit Patel | Correct | Correct | Correct |
| rahul Gandhi | Duplicate, not tested | Correct | Rejected |
| RDJ | Unknown | Correct | Unknown |
| Arpit Patel | Unknown | Retake | Unknown |
| Jony Deep | Retake | Retake | **Wrong: rahul Gandhi** |
| Krish Hemsworth | Rejected | Unknown | Retake |
| Tom Holland | Unknown | Unknown | Rejected |

---

## 2. LBP and HOG weights (80% of boxes kept)

"Current settings" uses cutoff 0.055, margin 0.004, and factor 1.1. "Held-out" chooses settings separately, as described above.

| LBP | HOG | AUC | EER | Closest | Correct | Wrong | Retake | Unknown | Strangers | Held-out correct | Held-out wrong | Held-out strangers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.0 | 0.672 | 39.7% | 9 | 4 | 0 | 0 | 13 | 1 | 3 | 1 | 2 |
| 0.1 | 0.9 | 0.680 | 34.8% | 9 | 4 | 0 | 0 | 13 | 0 | 5 | 0 | 2 |
| 0.2 | 0.8 | 0.686 | 35.3% | 9 | 5 | 0 | 1 | 11 | 0 | 5 | 0 | 0 |
| 0.3 | 0.7 | 0.702 | 35.3% | 9 | 5 | 0 | 1 | 11 | 0 | 5 | 0 | 0 |
| **0.4 (applied)** | **0.6 (applied)** | **0.712** | **35.3%** | **8** | **5** | **0** | **3** | **9** | **0** | **5** | **0** | **0** |
| 0.5 | 0.5 | 0.717 | 39.7% | 8 | 5 | 0 | 3 | 9 | 1 | 3 | 1 | 1 |
| 0.6 (previous) | 0.4 (previous) | 0.729 | 41.2% | 8 | 5 | 1 | 4 | 7 | 4 | 3 | 1 | 1 |
| 0.7 | 0.3 | 0.728 | 35.8% | 8 | 5 | 4 | 5 | 3 | 6 | 2 | 1 | 1 |
| 0.8 | 0.2 | 0.729 | 35.3% | 6 | 5 | 2 | 7 | 3 | 4 | 1 | 2 | 4 |
| 0.9 | 0.1 | 0.713 | 41.2% | 7 | 6 | 2 | 7 | 2 | 4 | 1 | 1 | 3 |
| 1.0 | 0.0 | 0.695 | 41.2% | 7 | 5 | 3 | 8 | 1 | 4 | 1 | 2 | 4 |

Counts are out of 17 scans, except held-out correct and wrong, which come from the same 17 scans with settings chosen separately.

## 3. Share of boxes kept

### LBP 0.6, HOG 0.4

| Boxes kept | AUC | EER | Closest | Correct | Wrong | Retake | Unknown | Strangers | Held-out correct | Held-out wrong | Held-out strangers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 (all 64) | 0.711 | 35.3% | 7 | 5 | 5 | 2 | 5 | 6 | 1 | 1 | 1 |
| 0.9 | 0.722 | 40.2% | 8 | 5 | 4 | 4 | 4 | 7 | 3 | 0 | 0 |
| 0.8 (kept) | 0.729 | 41.2% | 8 | 5 | 1 | 4 | 7 | 4 | 3 | 1 | 1 |
| 0.7 | 0.731 | 40.2% | 6 | 4 | 2 | 5 | 6 | 5 | 4 | 1 | 2 |
| 0.6 | 0.727 | 35.3% | 6 | 3 | 1 | 7 | 6 | 2 | 6 | 1 | 1 |
| 0.5 | 0.728 | 35.3% | 7 | 3 | 1 | 7 | 6 | 1 | 5 | 1 | 1 |

### LBP 0.3, HOG 0.7

| Boxes kept | AUC | EER | Closest | Correct | Wrong | Retake | Unknown | Strangers | Held-out correct | Held-out wrong | Held-out strangers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 (all 64) | 0.676 | 41.2% | 8 | 5 | 1 | 0 | 11 | 3 | 1 | 0 | 0 |
| 0.9 | 0.689 | 41.2% | 8 | 5 | 1 | 1 | 10 | 1 | 5 | 1 | 1 |
| 0.8 | 0.702 | 35.3% | 9 | 5 | 0 | 1 | 11 | 0 | 5 | 0 | 0 |
| 0.7 | 0.705 | 35.3% | 9 | 4 | 0 | 2 | 11 | 0 | 5 | 0 | 1 |
| 0.6 | 0.710 | 35.3% | 9 | 3 | 0 | 3 | 11 | 0 | 4 | 0 | 0 |
| 0.5 | 0.703 | 36.8% | 9 | 3 | 0 | 3 | 11 | 0 | 2 | 1 | 2 |

---

## 4. Weight change decision (applied)

| | LBP | HOG | Correct | Wrong | Strangers | Held-out correct / wrong / strangers |
|---|---:|---:|---:|---:|---:|---:|
| Previous | 0.6 | 0.4 | 5 | 1 | 4 | 3 / 1 / 1 |
| **Applied** | **0.4** | **0.6** | **5** | **0** | **0** | **5 / 0 / 0** |

**Why:** on these photos, changing the weights keeps 5 correct, and removes the wrong-person result and all 4 accepted strangers. The EER also improves from 41.2% to 35.3%.

**Trade-offs**
- **AUC falls slightly:** from 0.729 to 0.712, so overall separation isn't clearly better.
- **More Unknown results:** Unknown rises from 7 to 9, while Retake falls from 4 to 3.

**Other options**
- **LBP 0.3, HOG 0.7:** similar safety, with one more correct closest match, but 11 Unknown results.
- **Keeping fewer boxes:** this may reduce accepted strangers, but the results weren't consistent, so it needs more photos first.

## 5. Other findings

- **Weights can't fix everything:** correct verifications stayed at about 5 of 17 at nearly every weight, so many rejections come from the limits and the photos.
- **One registered photo attracts other people:** `rahul Gandhi`'s photo was the closest match for 7 of the 16 accepted scans of other people. Replacing it may reduce errors.
- **Some registered photos are very close:** Krish Hemsworth and RDJ are 0.027 apart, and Krish Hemsworth and rahul Gandhi are 0.030 apart, which makes their personal limits very tight.
- **Webcam photos did best:** Mit's scans were correct 3 of 3 times. Most other people's photos came from mixed web sources.

## 6. How far to trust these results

- **Small test:** only 17 usable photos of 7 people, so one photo changes a result by about 6%.
- **Same-photo tuning:** the applied weights were chosen on these same photos, so a real-world gain may be smaller, even with the held-out check.
- **Mixed photo sources:** most photos aren't from the app's webcam, so live results may differ.

---

## 7. Results after the change (LBP 0.4 / HOG 0.6)

Same 20 photos, same cutoff, margin, and personal limit factor. Measured using the app's own code and the real API on a temporary copy. Registered-person results come from the 20 API scans, and stranger results from the app's own comparison code.

| Result | Before (0.6 / 0.4) | After (0.4 / 0.6) |
|---|---:|---:|
| **Correct** | 5 of 20 | **5 of 20 (25%)** |
| **Wrong person verified** | 1 of 20 | **0 of 20** |
| Please retake | 4 | 3 |
| Unknown person | 7 | 9 |
| Photo rejected | 3 | 3 |
| **Strangers verified** | 4 of 17 | **0 of 17** |
| Strangers: Please retake | Not recorded separately | 3 of 17 |
| Strangers: Unknown person | Not recorded separately | 14 of 17 |
| Closest correct | 8 of 17 | 8 of 17 |
| AUC | 0.729 | 0.712 |
| EER | 41.2% | 35.3% |
| Latency, median | 46 ms | 49 ms |
| RAM, normal / peak | 102 / 113 MB | 101 / 112 MB |

### Per person after the change

| Person | Photo 1 | Photo 2 | Photo 3 |
|---|---|---|---|
| Mit Patel | Correct | Correct | Correct |
| rahul Gandhi | Duplicate, not tested | Correct | Rejected |
| RDJ | Unknown | Correct | Unknown |
| Arpit Patel | Unknown | Retake | Unknown |
| Jony Deep | Retake | Retake | Unknown |
| Krish Hemsworth | Rejected | Unknown | Unknown |
| Tom Holland | Unknown | Unknown | Rejected |
