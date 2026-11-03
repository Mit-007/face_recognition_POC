# Face Recognition POC: Approach Metrics

Approximate accuracy, latency, and RAM for every approach tried in this application. Figures come from tests run on 2026-11-02 on a Windows PC with 12 CPU threads.

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
| 6 | **YuNet + photo checks + worst boxes dropped + personal limit** | 9 | **0.575** | **50.0%** | **50%** | **37.5–40%** | **44–45%** | **About 75–95 ms** | **About 102 MB** | **About 112 MB** | **Current** |

\* Measured with OpenCL off and the low-memory server settings, on the Haar code that was current then.

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
| **YuNet, current formula** | **Not separately measured** | **102 MB** | **112 MB** | **About 150 MB** |

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
| Face numbers (LBP + HOG) | 4.5 KB raw, about 5 KB in Python | 0.5 MB | 5 MB |
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
| One comparison, current formula | About 0.05 ms | Test tool |

### Estimated comparison time as people increase (current formula)

The current formula also compares the closest person with everyone else for the personal limit, so a scan makes about 2 × N comparisons.

| Registered people | Comparison time | Approx. total per request |
|---:|---:|---:|
| 10 | 1 ms | 85 ms |
| 100 | 10 ms | 95 ms |
| 1,000 | 100 ms | 185 ms |
| 10,000 | 1,000 ms | About 1.1 s |

These are calculated estimates. The tests used only a handful of registered people.

---

## 9. Summary

| Goal | Best measured option | Trade-off |
|---|---|---|
| Lowest RAM with normal photos | Haar, OpenCL off (97 MB) | Could process fewer photos: 7 of 10 |
| Most photos processed | YuNet, no photo checks (10 of 10) | Poor photos aren't filtered |
| Best balance in the app today | Row 6 in the main table | Still accepts many sample strangers |
| Best separation on samples | `combined` (AUC 0.867) | About 36 KB per person, and needs more testing |

No approach here reliably separates right from wrong. Better results need more testing with labeled webcam photos, reference-face scoring, or an ID or PIN.
