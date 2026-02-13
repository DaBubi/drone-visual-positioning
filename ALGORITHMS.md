# Visual Positioning System — Algorithms & Solutions

GPS estimation from drone nadir imagery by matching against satellite images.

---

## System Architecture

```
Drone Camera (nadir) → Coarse Retrieval → Fine Matching → Homography → GPS Extraction
                            ↓                   ↓              ↓
                      NetVLAD/FAISS      SuperPoint+LightGlue  cv2.findHomography
                      Top-k tiles        Point correspondences  Pixel → Tile → LatLon
```

### Pipeline Stages

| Stage | Purpose | Method | Accuracy |
|-------|---------|--------|----------|
| 1. Coarse retrieval | Find candidate region | NetVLAD image descriptors + FAISS index | ~1-5 km |
| 2. Fine matching | Match drone to satellite tile | SuperPoint + LightGlue feature matching | ~5-20 m |
| 3. Homography | Compute geometric transform | RANSAC homography estimation | ~1-5 m |
| 4. GPS extraction | Convert pixel to coordinates | Tile coordinate math (Web Mercator) | sub-meter potential |
| 5. Fusion (optional) | Smooth trajectory | EKF with visual odometry + IMU | continuous |

---

## 1. Feature-Based Matching

### Traditional Features

**SIFT** (Scale-Invariant Feature Transform)
- 128-dim descriptors from gradient histograms at multiple scales
- Best accuracy for satellite stereo imagery (2024 evaluations)
- Slow but gold-standard reliability
- `cv2.SIFT_create()`

**ORB** (Oriented FAST and Rotated BRIEF)
- Real-time, patent-free alternative to SIFT
- Binary descriptors — much faster matching
- Less robust to large viewpoint changes
- `cv2.ORB_create()`

### Learned Features

**SuperPoint** — Self-supervised CNN for joint keypoint detection + description. Trained on synthetic homographic warps. Better cross-view robustness than hand-crafted features.

**LightGlue** (ICCV 2023) — Adaptive feature matcher with early stopping. 2x faster than SuperGlue, Apache 2.0 license, state-of-the-art on satellite imagery. Pairs with SuperPoint or DISK features.

**LoFTR** — Detector-free semi-dense matcher using transformers. Good for low-texture areas where keypoint detectors fail. Efficient LoFTR (CVPR 2024) achieves sparse-like speed.

### Recommendation
**SuperPoint + LightGlue** — best speed/accuracy tradeoff, permissive license, proven on satellite data.

---

## 2. Deep Learning Cross-View Geo-Localization

Models trained specifically to match drone/ground views to satellite views.

**GeoDTR+** (TPAMI 2024) — Geometric Disentanglement Transformer. Separates geometric layout from raw features. State-of-the-art on CVUSA/CVACT/VIGOR benchmarks.

**Sample4Geo** — Hard negative mining using geographical neighbors. Better training for visually similar locations.

**TransGeo** — Pure transformer architecture treating geo-localization as sequence-to-sequence.

**EP-BEV** (ECCV 2024) — Panorama-BEV co-retrieval. Leading results on VIGOR dataset.

**STHN** (RA-L 2024) — Coarse-to-fine deep homography specifically for UAV-to-satellite. Works with thermal imagery too. Reliable within 512m radius.

### Key Datasets
- **CVUSA**: 35k training pairs (North America)
- **CVACT**: 128k pairs (Australia)
- **VIGOR**: Cross-region generalization
- **University-1652**: 50k images, drone-to-satellite, 1652 buildings

---

## 3. Image Retrieval (Coarse Localization)

Find which satellite tile region the drone is over before doing expensive matching.

**NetVLAD** — CNN with VLAD pooling layer for compact global image descriptors. Fast retrieval with FAISS index.

**MS-NetVLAD** (2024) — Multi-scale extension, outperforms original on all benchmarks.

**SuperVLAD** (NeurIPS 2024) — Transformer backbone + VLAD, current state-of-the-art for visual place recognition.

**AnyLoc** — Foundation model features for universal place recognition. Best generalization across environments.

### Retrieval Pipeline
1. Extract global descriptor from drone image (~50ms)
2. Search pre-built FAISS index of satellite tile descriptors (~5ms)
3. Return top-k candidate tiles (k=5-50)
4. Filter by approximate location if available

---

## 4. Homography Estimation

Given matched point pairs between drone and satellite images, compute the geometric transform.

**RANSAC + DLT** — Standard approach. Iteratively fits homography from 4-point samples, rejects outliers.
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, confidence=0.999)
inlier_ratio = np.sum(mask) / len(mask)  # Confidence metric
```

**Deep Homography** — CNN regression predicting 4-corner offsets directly. End-to-end differentiable. STHN uses coarse-to-fine deep homography for UAV-satellite specifically.

### GPS from Homography
```python
# Transform drone image center through homography
center = np.array([[w/2, h/2]], dtype=np.float32)
sat_point = cv2.perspectiveTransform(center.reshape(1,1,2), H)

# Convert satellite pixel to GPS via tile coordinates
n = 2.0 ** zoom
lon = (tile_x + sat_x / 256.0) / n * 360.0 - 180.0
lat = degrees(atan(sinh(pi * (1 - 2 * (tile_y + sat_y / 256.0) / n))))
```

---

## 5. Correlation-Based Methods

**Normalized Cross-Correlation (NCC)** — Slide drone image over satellite tile, find peak correlation. Simple, no feature extraction needed. Works for small search areas with similar viewpoints. `cv2.matchTemplate(method=cv2.TM_CCOEFF_NORMED)`

**Phase Correlation** — FFT-based translational offset detection. Very fast (O(n log n)), sub-pixel accuracy. Only handles translation — needs extensions for rotation/scale. `skimage.registration.phase_cross_correlation()`

Best used as a refinement step after coarse localization narrows the search area.

---

## 6. Hierarchical / Multi-Scale Approaches

Real systems combine multiple methods in a coarse-to-fine pipeline:

**Stage 1 — Global** (1-5 km): NetVLAD retrieval on low-res (512x512) imagery. Search pre-built database. ~100ms.

**Stage 2 — Local** (100-500 m): Dense feature matching (LightGlue) on medium-res tiles. Filter to top-3 candidates. ~200ms.

**Stage 3 — Precise** (<10 m): High-res homography estimation + sub-pixel refinement. Use zoom 19 tiles (0.3 m/pixel). ~100ms.

### 2024-2025 Advances
- Hierarchical distillation: knowledge distillation from large models to edge-deployable ones
- Lightweight algorithms optimized for UAV edge computing (Jetson, RPi)
- Multi-scale feature fusion networks (MFRGN, MCFA)

---

## 7. Visual Odometry (Relative Tracking)

Between absolute satellite matches, track relative motion for continuous positioning.

**ORB-SLAM3** — Feature-based SLAM with visual-inertial fusion. Runs on Jetson Nano (3-11cm accuracy on benchmarks).

**VINS-Fusion** — Monocular/stereo VIO with loop closure. Robust to challenging conditions.

**RAFT** — Deep optical flow for dense motion estimation between frames.

### Fusion Pattern
```
VIO provides: continuous relative pose (drifts over time)
Satellite matching provides: absolute position (periodic, ~every 5-10s)
EKF combines: smooth, drift-corrected trajectory
```

---

## 8. Satellite Tile Management

### Tile Sources
| Provider | Endpoint | Cost | Resolution |
|----------|----------|------|------------|
| Mapbox Satellite | `api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png` | Free tier + usage | High |
| Google Maps | `tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}` | API key + pricing | Very high |
| OpenStreetMap | `tile.openstreetmap.org/{z}/{x}/{y}.png` | Free (usage policy) | Medium |
| Sentinel-2 | Copernicus API | Free | 10m/pixel |

### Tile Coordinate System (Web Mercator)
- Zoom z → 2^z × 2^z tiles, each 256×256 px
- Zoom 15: ~4.77 m/pixel | Zoom 17: ~1.19 m/pixel | Zoom 19: ~0.30 m/pixel

### Caching Strategy
- Pre-download area of operations at zoom 15, 17, 19
- Store as `cache/{provider}/{z}/{x}/{y}.png` or MBTiles (SQLite)
- Build FAISS index of NetVLAD descriptors per tile
- LRU eviction for memory limits

### Python Libraries
- `pygeotile` — tile/GPS coordinate conversions
- `pyproj` — CRS transformations (WGS84 ↔ Web Mercator)
- `rasterio` — GeoTIFF reading
- `requests` — tile downloading

---

## 9. Coordinate Math

### GPS ↔ Tile Conversion
```python
import math

def gps_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def tile_pixel_to_gps(tile_x, tile_y, zoom, px, py):
    n = 2.0 ** zoom
    lon = (tile_x + px / 256.0) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + py / 256.0) / n))))
    return lat, lon
```

### Accuracy at Different Zoom Levels
| Zoom | m/pixel (equator) | 1-pixel error |
|------|-------------------|---------------|
| 15 | 4.77 m | ~5 m |
| 17 | 1.19 m | ~1.2 m |
| 19 | 0.30 m | ~0.3 m |
| 21 | 0.07 m | ~0.07 m |

---

## 10. Open-Source References

| Project | Description | URL |
|---------|-------------|-----|
| **Wildnav** | GNSS-free drone navigation via satellite matching | `TIERS/wildnav` |
| **STHN** | Deep homography for UAV thermal geo-localization | `arplaboratory/STHN` |
| **LightGlue** | Fast feature matcher (SuperPoint+LightGlue) | `cvg/LightGlue` |
| **Image Matching Models** | Unified API for 50+ matchers | `gmberton/image-matching-models` |
| **OpenAthena** | Drone target geo-localization from metadata | `Theta-Limited/OpenAthena` |
| **GeoDTR+** | Cross-view geo-localization transformer | `zxh009123/GeoDTR_plus` |
| **EP-BEV** | Panorama-BEV cross-view retrieval | `yejy53/EP-BEV` |
| **Awesome Cross-View** | Curated paper/code list | `GDAOSU/Awesome-Cross-View-Methods` |

---

## Recommended Implementation Stack

```
opencv-python>=4.8       # Feature matching, homography, image processing
torch>=2.0               # Deep learning backend
kornia>=0.7              # SuperPoint, LightGlue (PyTorch CV)
faiss-cpu>=1.7           # Fast similarity search for tile retrieval
pygeotile>=1.0           # Tile coordinate math
pyproj>=3.5              # Coordinate transformations
rasterio>=1.3            # GeoTIFF handling
requests>=2.31           # Tile downloads
numpy, pillow            # Core utilities
flask>=3.0               # Dashboard / API
```

## Expected Accuracy

| Condition | Coarse | Fine | Precision |
|-----------|--------|------|-----------|
| Urban, high texture | 100-500 m | 5-20 m | 1-5 m |
| Suburban | 200-1000 m | 10-50 m | 3-10 m |
| Rural, low texture | 500-2000 m | 20-100 m | 5-20 m |
| Featureless (water, desert) | May fail | May fail | N/A |
