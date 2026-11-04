# Face Recognition POC: Approach Metrics

Approximate accuracy, latency, and RAM for every approach tried in this application. Figures come from tests run on 2026-11-02 and 2026-11-03 on a Windows PC with 12 CPU threads.

**Current approach (2026-11-03): wider crop, blur, LBP radius 2, 8 px HOG, and relative distance with the scan average, row 8 and section 11.** Sections 2–5 and 7–8 record earlier work unless a row says otherwise.

## How to read these numbers

- **Small test set:** accuracy comes from 10 sample web and phone photos of 5 people, plus webcam portraits in some tests. Differences of one or two photos can change a result a lot, so treat these as rough comparisons, not real-world rates.
- **Not always the same photos:** approaches that reject more photos are tested on fewer, easier photos. The photos used are listed for each test.
- **Live webcam results may differ:** the sample photos are harder than scans from one camera in one place.
- **"Not measured"** means no reliable figure exists for that item.

| Term | Meaning | Better when |
|---|---|---|
| **AUC** (separation score) | Chance that a same-person pair is closer than a different-people pair. 0.5 is guessing, and 1.0 is perfect | Higher |
| **EER** | Error rate where wrong accepts and wrong rejects are equal | Lower |
| **Closest correct** | The closest registered person is the right person, with no limit applied | Higher |
| **Known verified** | A registered person gets verified as themselves | Higher |
| **Stranger accepted** | An unregistered person gets verified as someone | Lower |
| **Normal / peak RAM** | Whole backend process, with photos up to about 640 px | Lower |

---

## 1. Main approaches

| # | Approach | Photos used | AUC | EER | Closest correct | Known verified | Stranger accepted | Latency per request | Normal RAM | Peak RAM | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | YuNet + SFace embeddings + Pinecone | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | About 1,400 ms (one old log record) | Not measured | Not measured | Removed: embeddings and cloud database |
| 2 | Haar crop + LBP + HOG, cutoff 0.22 | 10 | 0.550 | 60.6% | 40% | 40% | 100% | About 120–150 ms | About 97 MB* | About 100 MB* | Replaced |
| 3 | Haar + eye alignment | 7 | 0.250 | 58.3% | 0% | Not measured | Not measured | About 120–150 ms | About 97 MB* | About 100 MB* | Replaced |
| 4 | YuNet + landmark alignment, no photo checks | 10 | 0.517 | 60.6% | 30% | Not measured | Not measured | About 75–85 ms | About 102 MB | About 114 MB | Replaced |
| 5 | YuNet + photo checks, whole-face formula | 9 | 0.558 | 50.0% | 37.5% | 30% (tuning test) | 64% (tuning test) | About 75–85 ms | About 102 MB | About 114 MB | Replaced |
| 6 | YuNet + photo checks + worst boxes dropped + personal limit | 9 | 0.575 | 50.0% | 50% | 37.5–40% | 44–45% | About 75–95 ms | About 102 MB | About 112 MB | Replaced |
| 7 | Row 6 with 8 px HOG + relative distance, no personal limit | 17 (different set) | 0.858 | 18.1% | 65% | 41% | 6% | 36–75 ms | About 101 MB | About 113 MB | Replaced |
| 8 | **Row 7 with wider crop, blur, LBP radius 2, and scan average** | **17 (different set)** | **0.934** | **11.8%** | **71%** | **59%** | **0%, but see section 11** | **About 30 ms** | **About 101 MB** | **About 112 MB** | **Current** |

\* Measured with OpenCL off and the low-memory server settings, on the Haar code that was current then.

Rows 7 and 8 use the `D:\face_test` photos, so don't compare them directly with the rows above. Sections 10 and 11 compare them on the same photos.

**Notes**
- Row 3 used the 7 photos Haar could process, so its results can't be compared directly with the others. It has only 2 same-person pairs.
- Rows 5 and 6 use the same 9 photos for AUC, EER, and closest correct. For "known verified" and "stranger accepted", row 5 uses 3 of 10 and 7 of 11, and row 6 uses both a 9-photo test (3 of 8 and 4 of 9) and an 11-photo tuning test (4 of 10 and 5 of 11).
- Row 6 settings: cutoff 0.055, margin 0.004, personal limit factor 1.1.
- With only one registered person, the margin and personal limit can't apply, and 3 of 8 sample strangers were verified.

---

## 2. Feature options tested on YuNet

These use the 9 photos accepted by YuNet with photo checks. "Unknown accepted" uses a limit tuned separately for each option. Except for `current_app`, these options weren't added to the app.

| Option | AUC | EER | Closest correct | Unknown accepted | Time per photo | Data per person |
|---|---:|---:|---:|---:|---:|---:|
| `whole_face_previous` | 0.558 | 50.0% | 37.5% | 77.8% | About 33 ms | 4.5 KB |
| `current_app` | 0.575 | 50.0% | 50.0% | 77.8% | About 34 ms | 4.5 KB |
| `lbp_only` | 0.542 | 50.0% | 37.5% | 77.8% | About 35 ms | 4.5 KB |
| `hog_only` | 0.708 | 45.0% | 12.5% | 88.9% | About 31 ms | 4.5 KB |
| `inner_crop_80` | 0.650 | 46.7% | 25.0% | 77.8% | About 31 ms | 4.5 KB |
| `multiscale_lbp` | 0.717 | 30.8% | 37.5% | 66.7% | About 43 ms | 9.5 KB |
| `region_weighted` | 0.575 | 50.0% | 37.5% | 77.8% | About 32 ms | 4.5 KB |
| `finer_hog` | 0.683 | 50.0% | 25.0% | 77.8% | About 32 ms | 30.6 KB |
| `combined` | 0.867 | 24.2% | 62.5% | 66.7% | About 41 ms | 35.6 KB |

`whole_face_previous` and `current_app` were measured in a later run. The other rows were measured before the worst-box change, so they use the whole-face comparison style of that time. `combined` looks most promising, but it needs testing on more photos.

---

## 3. Distance formulas compared

Tested on 11 photos of 6 people. "Best tuned" is the most known people verified with at most one-third of strangers accepted, out of 10 known and 11 stranger scans.

| Formula | AUC | EER | Closest correct | Best tuned: known | Best tuned: strangers |
|---|---:|---:|---:|---:|---:|
| Chi-square + cosine, 0.6/0.4 (previous) | 0.530 | 60% | 3 of 10 | 2 | 1 |
| Chi-square + cosine, 0.4/0.6 | 0.587 | 59% | 3 of 10 | 2 | 1 |
| Hellinger + cosine, 0.6/0.4 | 0.548 | 60% | 3 of 10 | 2 | 1 |
| Hellinger + cosine, 0.4/0.6 | 0.604 | 59% | 3 of 10 | 3 | 3 |
| Chi-square, worst 20% dropped + whole cosine | 0.587 | 56% | 5 of 10 | 4 | 1 |
| **Chi-square + per-box cosine, worst 20% dropped (current)** | **0.583** | **56%** | **5 of 10** | **4** | **1** |
| Hellinger, worst 20% dropped + whole cosine | 0.574 | 55% | 5 of 10 | 4 | 1 |
| Hellinger + per-box cosine, worst 20% dropped | 0.578 | 55% | 6 of 10 | 4 | 1 |
| Per-box cosine only, worst 20% dropped | 0.739 | 22% | 4 of 10 | 2 | 2 |

The current app uses a less strict setting than "best tuned", which accepts more strangers to reduce rejections.

---

## 4. Cutoff and margin settings

Sample photos against the people registered at the time of each test. The registered people changed between tests, so compare rows within one test only.

| Test | Formula | Cutoff | Margin | Personal limit | Known verified | Strangers accepted |
|---|---|---:|---:|---:|---:|---:|
| A | Whole-face | 0.22 | 0 | None | 3 of 10 | 13 of 13 |
| A | Whole-face | 0.07 | 0.015 | None | 1 of 10 | 4 of 13 |
| A | Whole-face | 0.06 | 0.015 | None | 1 of 10 | 3 of 13 |
| B | Whole-face | 0.07 | 0.015 | None | 1 of 10 | 4 of 12 |
| B | Whole-face | 0.08 | 0.007 | None | 2 of 10 | 7 of 12 |
| B | Whole-face | 0.07 | 0.004 | None | 3 of 10 | 7 of 12 |
| B | Whole-face | 0.085 | 0.002 | None | 3 of 10 | 12 of 12 |
| C | Whole-face | 0.07 | 0.004 | None | 3 of 10 | 7 of 11 |
| C | Worst boxes dropped | 0.055 | 0.004 | None | 4 of 10 | 8 of 11 |
| C | Worst boxes dropped | 0.06 | 0.006 | 0.9 | 2 of 10 | 2 of 11 |
| C | Worst boxes dropped | 0.06 | 0.004 | 0.9 | 2 of 10 | 3 of 11 |
| C | **Worst boxes dropped (current)** | **0.055** | **0.004** | **1.1** | **4 of 10** | **5 of 11** |

---

## 5. Decision rules compared

Whole-face formula on 10 known and 11 stranger scans. Gap and ratio rows use a cutoff of 0.07. The first row gives 2 known verified here but 3 in section 4, because this test used unrounded distances.

| Rule | Known verified | Strangers accepted | Used? |
|---|---:|---:|---|
| Gap of at least 0.004 | 2 | 7 | Yes |
| Ratio up to 0.95 | 3 | 8 | No |
| Ratio up to 0.90 | 2 | 7 | No |
| Ratio up to 0.85 | 2 | 6 | No |
| Personal limit × 1.1 | 3 | 7 | Yes, combined with the gap |
| Personal limit × 1.0 | 2 | 6 | No |
| Personal limit × 0.9 | 2 | 3 | No, stricter option |
| Gap + HOG limit 0.06 | 2 | 4 | No |
| Gap + HOG limit 0.10 | 2 | 7 | No |

Personal limit rows use only the personal limit, with no cutoff or gap. "Gap + HOG limit" rows also require the HOG distance to be within that limit.

---

## 6. RAM by configuration

Whole backend process on Windows.

| Configuration | After startup | Normal use | Peak, normal photos | Peak, 2448×3264 photo |
|---|---:|---:|---:|---:|
| Haar, OpenCL on, default server | 81 MB | 211 MB | Not measured | 490 MB |
| Haar, OpenCL off, low-memory server | 81 MB | 97 MB | 100 MB | 378 MB |
| YuNet, whole-face formula | 81 MB | 102 MB | 114 MB | 150 MB |
| YuNet, worst boxes dropped, 16 px HOG | Not separately measured | 102 MB | 112 MB | About 150 MB |
| YuNet, 8 px HOG + relative distance | 81 MB | 101 MB | 113 MB | Not measured |
| **YuNet, row 8 (current)** | **81 MB** | **101 MB** | **112 MB** | **Not measured** |

Low-memory server command:

```powershell
uvicorn main:app --port 8000 --http h11 --ws none --loop asyncio
```

Inside an existing Python process that has already loaded Python, NumPy, and the server libraries, the app's own use should be about 48 MB lower: about 54 MB in normal use and 64–66 MB at peak. This is an estimate, not a measurement.

---

## 7. RAM by part (approximate)

| Part | RAM | Basis |
|---|---:|---|
| Python | 15 MB | Measured |
| NumPy | 10 MB | Measured |
| OpenCV, with OpenCL off | 8 MB | Measured |
| FastAPI, Pydantic, Starlette, uvicorn, multipart | 23 MB | Measured together |
| `scikit-image`, face service code, and related code | 20 MB | Measured together |
| YuNet model and other startup memory | 5 MB | Remainder |
| **After startup** | **About 81 MB** | **Measured on the server** |
| YuNet detection working memory | 10–15 MB | Estimate |
| Web requests and other internal buffers | 5–10 MB | Remainder |
| **Normal use** | **About 102 MB** | **Measured on the server** |
| Short spike while processing a photo | About 10–12 MB | Measured on the server |
| **Peak** | **About 112–114 MB** | **Measured on the server** |
| OpenCL, if turned on | About 90–110 MB more | Measured in a test process |

### Stored data

| Data | Size per item | 100 people | 1,000 people |
|---|---:|---:|---:|
| Face numbers (LBP + 8 px HOG) | 10.5 KB raw, about 11 KB in Python | 1.1 MB | 11 MB |
| User records | About 1 KB | 0.1 MB | 1 MB |
| Results list during a scan | About 0.5 KB | 0.05 MB | 0.5 MB |
| Detection history (500 records at most) | About 1 KB | Up to 0.5 MB | Up to 0.5 MB |

Saved portraits and JSON files use disk, not RAM, except briefly when face numbers are recalculated after a restart.

---

## 8. Latency

| Step | Approximate time | Basis |
|---|---:|---|
| Full request, Haar, 640×480 webcam photo | 120–150 ms | Logged scans |
| Full request, YuNet, 640×480 webcam photo | 75–95 ms | API test |
| Full request, YuNet, 2448×3264 photo | About 210 ms | API test |
| First request after a restart, 4 people | About 200 ms | API test |
| Process one photo in YuNet + LBP + HOG, sample photos | About 33–41 ms | Test tool |
| One comparison, whole-face formula | About 0.014 ms | Test tool |
| One comparison, worst boxes dropped, 16 px HOG | About 0.03–0.05 ms | Test tool |
| One comparison, current formula | About 0.04 ms | Test tool |
| Full request, current approach, `D:\face_test` photos | About 75 ms median | API test |

### Estimated comparison time as people increase (current formula)

A scan compares the photo with each of the N registered people. After someone registers or is removed, the next scan also recalculates all averages, which takes N × (N − 1) / 2 comparisons.

| Registered people | Comparison time per scan | Approx. total per request | One-time recalculation |
|---:|---:|---:|---:|
| 10 | 0.4 ms | 75 ms | 2 ms |
| 100 | 4 ms | 80 ms | 0.2 s |
| 1,000 | 40 ms | 115 ms | 20 s |
| 10,000 | 400 ms | About 0.5 s | About 33 minutes |

These are calculated estimates. The tests used only a handful of registered people.

---

## 9. Summary

| Goal | Best measured option | Trade-off |
|---|---|---|
| Lowest RAM with normal photos | Haar, OpenCL off (97 MB) | Could process fewer photos: 7 of 10 |
| Most photos processed | YuNet, no photo checks (10 of 10) | Poor photos aren't filtered |
| Best balance in the app today | Row 8 in the main table | Accepts more strangers when registration photos vary, and needs testing on new photos |
| Best separation on samples | `combined` (AUC 0.867) | About 36 KB per person, and needs more testing |

No approach here reliably separates right from wrong. Better results need more testing with labeled webcam photos, reference-face scoring, or an ID or PIN.

---

## 10. 8 px HOG + relative distance (replaced by section 11)

Tested on 2026-11-03 on `D:\face_test`: 7 registered people and 20 test photos. `rahul Gandhi/1` is excluded because it's the same file as the registration photo. Three photos were rejected for shadow, both before and after, leaving 17. Stranger results remove the person from the registered list and scan their photo again.

**What changed**
- HOG boxes went from 16 px to 8 px: 512 to 2,048 numbers, and 4.5 KB to 10.5 KB per person.
- `Relative distance = distance / that person's average distance to the other registered people`. It's used for ranking, the limit, and the gap, and replaces the personal limit.
- Settings: `MATCH_RELATIVE_LIMIT` 0.78, `MATCH_CLEAR_WINNER_MARGIN` 0.04, and `MATCH_DISTANCE_CUTOFF` 0.075, which is used only with one person registered.

**Before and after, same photos**

| Measure | Before | After |
|---|---:|---:|
| AUC | 0.712 | 0.858 |
| EER | 35.3% | 18.1% |
| Closest correct | 8 of 17 | 11 of 17 |
| Correct person verified | 5 of 20 | 7 of 20 |
| Wrong person verified | 0 of 20 | 0 of 20 |
| Please retake | 3 of 20 | 0 of 20 |
| Unknown, although registered | 9 of 20 | 10 of 20 |
| Rejected by photo checks | 3 of 20 | 3 of 20 |
| Strangers verified | 0 of 17 | 1 of 17 |
| Strangers given "Please retake" | 3 of 17 | 0 of 17 |
| Normal / peak RAM, whole server | About 102 / 112 MB | 101 / 113 MB |

**Choosing the limit and gap (same 17 photos)**

| Limit | Gap | Correct | Wrong | Retake | Unknown | Strangers verified | Strangers retake |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.72 | 0.04 | 3 | 0 | 0 | 14 | 0 | 0 |
| 0.76 | 0.04 | 5 | 0 | 0 | 12 | 1 | 0 |
| **0.78** | **0.04** | **7** | **0** | **0** | **10** | **1** | **0** |
| 0.80 | 0.00 | 8 | 0 | 0 | 9 | 2 | 0 |
| 0.84 | 0.04 | 9 | 0 | 2 | 6 | 3 | 2 |
| 0.88 | 0.04 | 9 | 0 | 5 | 3 | 5 | 3 |
| 0.92 | 0.04 | 10 | 0 | 6 | 1 | 7 | 5 |

**Held-out check:** settings were chosen on six people and scored on the seventh, repeated for each person. That gave 7 correct, 0 wrong, 1 retake, and 9 unknown out of 17, with 2 of 17 strangers verified.

**Test tool, which registers each person's `1.jpg` instead of their registered photo and includes the duplicate Rahul photo**

| Measure | Before | After |
|---|---:|---:|
| Closest correct | 45.8% | 66.7% |
| Known verified | 33.3% | 33.3% |
| Unknown accepted | 29.2% | 4.2% |

**Limits of this test**
- The limit was tuned on these same 17 photos, and this option was the best of 48 tried, so results on new photos may be lower.
- Most correct results come from a few people: all 3 `mit` photos passed, but `arpit`, `rahul Gandhi`, and `tom` had none. `rahul Gandhi/2` was verified before and is now Unknown, at 0.796 against the 0.78 limit.
- With one person registered, the app can only use plain distance. At cutoff 0.075, about 47% of same-person and 10% of different-person pairs were accepted.
- Scores depend on who is registered, so adding or removing someone slightly changes everyone's score.

---

## 11. Wider crop, blur, LBP radius 2, and scan average (current)

Tested on 2026-11-03. About 70 combinations were tried on `D:\face_test`, with registration photos rotated across trials, and `data_images` was kept as a separate check.

**What changed**
- Matching uses a slightly wider crop, with the landmark template scaled by 0.9. The covering checks keep the original crop.
- A Gaussian blur (sigma 1.0) is applied after CLAHE.
- LBP uses radius 2 instead of 1.
- `Relative distance = distance / that person's average distance to the others / this scan's average`.
- Settings: `MATCH_RELATIVE_LIMIT` 0.90, `MATCH_CLEAR_WINNER_MARGIN` 0.02, and `MATCH_DISTANCE_CUTOFF` 0.12, which is used only with one person registered.

**Fixed test: the app's 7 registered photos, 20 test photos, 3 rejected for shadow.** Measured through the real API on a temporary copy, except AUC, EER, and strangers, which use the app's own ranking code.

| Measure | Section 10 | Current |
|---|---:|---:|
| AUC | 0.858 | 0.934 |
| EER | 18.1% | 11.8% |
| Closest correct | 11 of 17 | 12 of 17 |
| Correct person verified | 7 of 20 | 10 of 20 |
| Wrong person verified | 0 of 20 | 0 of 20 |
| Please retake | 0 of 20 | 1 of 20 |
| Unknown, although registered | 10 of 20 | 6 of 20 |
| Rejected by photo checks | 3 of 20 | 3 of 20 |
| Strangers verified | 1 of 17 | 0 of 17 |
| Strangers given "Please retake" | 0 of 17 | 1 of 17 |
| Normal / peak RAM, whole server | 101 / 113 MB | 101 / 112 MB |

**The fixed test flatters the 0.90 limit.** It was tuned on those same registered photos. In this test, each person's registration photo changes across 500 trials while the limit stays fixed, so the stranger rate is much higher.

| Settings, `D:\face_test` (24 photos) | Correct | Wrong | Retake | Unknown | Strangers verified | Strangers retake |
|---|---:|---:|---:|---:|---:|---:|
| Section 10: limit 0.78, gap 0.04 | 32.5% | 1.8% | 3.7% | 62.0% | 5.1% | 1.9% |
| Current code, limit 0.84, gap 0.02 | 34.2% | 0.3% | 0.0% | 65.6% | 0.6% | 0.0% |
| Current code, limit 0.86, gap 0.02 | 38.2% | 1.0% | 0.1% | 60.7% | 2.3% | 0.0% |
| Current code, limit 0.88, gap 0.02 | 46.8% | 3.0% | 0.6% | 49.5% | 7.0% | 0.2% |
| **Current code, limit 0.90, gap 0.02 (applied)** | **57.2%** | **6.7%** | **2.3%** | **33.8%** | **18.3%** | **1.1%** |
| Current code, limit 0.92, gap 0.02 | 63.2% | 11.3% | 6.1% | 19.4% | 36.6% | 5.6% |

On the 8 `data_images` photos, the same test gives limit 0.78 (section 10): 13.3% correct, 11.5% wrong, 21.1% strangers. Limit 0.90 gives 30.7% correct, 17.5% wrong, 27.9% strangers. Limit 0.84 gives 17.5% correct, 0% wrong, 4.0% strangers. One photo changes these a lot.

**Test tool, which registers each person's `1.jpg` and includes the duplicate Rahul photo:** known verified went from 33.3% to 54.2%, closest correct from 66.7% to 70.8%, and unknown accepted from 4.2% to 33.3%. Its other pair figures use plain distance without the scan average, so they don't reflect the app's decision.

**Limits of this test**
- The result was the best of about 70 combinations on a small set of photos, so results on new photos may be lower.
- The rotating test includes poor registration photos that the app's stricter registration checks might refuse, so it's harder than real use. The truth is probably between the two tests.
- The scan average can't work with one person registered, and it has little to compare with two.
- With one person registered, at cutoff 0.12, about 53% of same-person and 8% of different-person pairs were accepted.
- Latency varies between runs: this code measured about 30 ms, and the section 10 code 36–75 ms. The added blur doesn't make it measurably slower.
