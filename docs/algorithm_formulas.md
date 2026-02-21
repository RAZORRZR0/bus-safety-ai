# Algorithm Formulas — Transport Policing System

> **Document Purpose**: Mathematical formulations of all algorithms implemented in the system, with references to source code.
> 
> **Project**: Deep Learning and Computer Vision Enabled Transport Policing System

---

## Table of Contents

1. [Violence Classifier — Temporal Mean-Pooling CNN](#1-violence-classifier--temporal-mean-pooling-cnn)
2. [Violence Confidence Scoring](#2-violence-confidence-scoring)
3. [Motion-Heuristic Aggression Estimator (Fallback)](#3-motion-heuristic-aggression-estimator-fallback)
4. [Trust Score Algorithm](#4-trust-score-algorithm)
5. [Severity Classification](#5-severity-classification)
6. [ROI-based Seat Violation Detection](#6-roi-based-seat-violation-detection)
7. [SHA-256 Hash Chain (Audit Trail)](#7-sha-256-hash-chain-audit-trail)
8. [Evidence Integrity Hashing](#8-evidence-integrity-hashing)
9. [YOLO Person Detection](#9-yolo-person-detection)
10. [Password Hashing (Authentication)](#10-password-hashing-authentication)

---

## 1. Violence Classifier — Temporal Mean-Pooling CNN

### Architecture

| Layer | Operation | Input Shape | Output Shape |
|-------|-----------|-------------|--------------|
| Input | Video clip of T frames | B × T × C × H × W | — |
| Reshape | Flatten batch & time | B × T × C × H × W | (B·T) × C × H × W |
| MobileNetV2 | Feature extraction | (B·T) × 3 × 112 × 112 | (B·T) × 1280 × 4 × 4 |
| AvgPool | Global average pooling | (B·T) × 1280 × 4 × 4 | (B·T) × 1280 |
| Reshape | Recover temporal dim | (B·T) × 1280 | B × T × 1280 |
| Mean-Pool | Average across time | B × T × 1280 | B × 1280 |
| Dropout | p = 0.3 | B × 1280 | B × 1280 |
| Linear | Classifier head | B × 1280 | B × 1 |
| Sigmoid | Output probability | B × 1 | B × 1 |

### Formal Equations

**Step 1 — Feature Extraction** (per frame):

```
fₜ = GlobalAvgPool( MobileNetV2_features(xₜ) )     where fₜ ∈ ℝ¹²⁸⁰
```

**Step 2 — Temporal Mean Pooling** (across T frames):

```
z = (1/T) × Σₜ₌₁ᵀ fₜ     where z ∈ ℝ¹²⁸⁰
```

**Step 3 — Classification**:

```
logit = W · Dropout(z, p=0.3) + b     where W ∈ ℝ¹ˣ¹²⁸⁰, b ∈ ℝ¹

ŷ = σ(logit) = 1 / (1 + e⁻ˡᵒᵍⁱᵗ)     where ŷ ∈ [0, 1]
```

### Training Loss — Binary Cross-Entropy

```
L_BCE = -(1/N) × Σᵢ₌₁ᴺ [ yᵢ · log(σ(ŷᵢ)) + (1 - yᵢ) · log(1 - σ(ŷᵢ)) ]
```

Where:
- `yᵢ ∈ {0, 1}` — ground truth label (0 = non-violence, 1 = violence)
- `ŷᵢ` — predicted logit
- `σ` — sigmoid function
- `N` — batch size

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1 × 10⁻⁴ |
| Scheduler | StepLR (step=5, γ=0.5) |
| Epochs | 12 |
| Batch Size | 8 |
| Frames per clip (T) | 16 |
| Frame resolution | 112 × 112 px |
| Dropout rate | 0.3 |
| ImageNet Normalization | μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225] |

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Model architecture | `app/services/inference.py` | 31–46 |
| Training loop | `train_violence.py` | 125–256 |
| Frame preprocessing | `app/services/inference.py` | 190–198 |
| Inference function | `app/services/inference.py` | 200–214 |

---

## 2. Violence Confidence Scoring

### Formula

```
confidence = σ(logit) = 1 / (1 + e⁻ˡᵒᵍⁱᵗ)

violence_detected = { True,   if confidence ≥ θ
                    { False,  if confidence < θ

where θ = 0.6 (configurable threshold)
```

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Sigmoid + threshold | `app/services/inference.py` | 210–213 |
| Video-level detection | `app/services/inference.py` | 237–243 |
| Threshold config | `app/core/config.py` | 19 |

---

## 3. Motion-Heuristic Aggression Estimator (Fallback)

> Used when the trained violence classifier model is unavailable.

### Formula

```
D = |F_current - F_previous|         (pixel-wise absolute difference)

motion = mean(D) / 255               (normalize to [0, 1])

aggression_score = min(1.0, motion × 2.4)    (amplify and clamp)
```

Where:
- `F_current`, `F_previous` — consecutive video frames as BGR matrices
- `D` — absolute difference matrix
- `2.4` — amplification constant (tuned to map typical motion magnitudes to [0, 1] range)

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Estimator function | `app/services/inference.py` | 161–167 |
| Fallback trigger | `app/services/inference.py` | 204–208 |

---

## 4. Trust Score Algorithm

### Formula

Each scope entity (bus, route, or timeslot) maintains an independent trust score:

```
Trust₀ = 100.0                       (initial score for all entities)

Penalty(event) = Σ of applicable:
    +3.0    if seat_misuse = True
    +15.0   if violence = True
    +10.0   if severity = "critical"

Trust_new = clamp(Trust_old - Penalty, 0, 100)

where clamp(x, a, b) = max(a, min(b, x))
```

### Scope Application

For each violation event, the penalty is applied independently to **three scopes**:

```
Trust_bus(bus_id)         = clamp(Trust_bus(bus_id) - Penalty, 0, 100)
Trust_route(route_id)     = clamp(Trust_route(route_id) - Penalty, 0, 100)
Trust_timeslot(timeslot)  = clamp(Trust_timeslot(timeslot) - Penalty, 0, 100)
```

### Penalty Table

| Condition | Penalty Points | Rationale |
|-----------|---------------|-----------|
| Seat misuse | -3.0 | Minor but frequent violation |
| Violence | -15.0 | Severe safety threat |
| Critical severity | -10.0 (additional) | Compounding for extreme events |
| Maximum single event | -28.0 | Seat misuse + violence + critical |

### Trust Score Decay Example

```
Event 1: seat_misuse only            → 100.0 - 3.0  = 97.0
Event 2: violence (high)             → 97.0  - 15.0 = 82.0
Event 3: violence + critical         → 82.0  - 25.0 = 57.0
Event 4: seat_misuse + violence + critical → 57.0 - 28.0 = 29.0
```

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Penalty calculation | `app/services/trust.py` | 8–16 |
| Score update (clamp) | `app/services/trust.py` | 19–29 |
| Tri-scope application | `app/services/trust.py` | 32–38 |

---

## 5. Severity Classification

### Decision Tree Formula

```
severity(violence, aggression_score, seat_misuse) =
    "critical"  if violence = True AND aggression_score ≥ 0.9
    "high"      if violence = True OR (seat_misuse = True AND aggression_score ≥ 0.5)
    "medium"    if seat_misuse = True
    "low"       otherwise
```

### Prerequisite Conditions

```
seat_misuse = (reserved_seats_occupied > 0)
            AND (special_category_present = False)
            AND (reserved_seats_total > 0)

violence = (aggression_score ≥ 0.75)
```

### Decision Matrix

| Violence | Aggression ≥ 0.9 | Seat Misuse | Aggression ≥ 0.5 | Severity |
|----------|-------------------|-------------|-------------------|----------|
| ✓ | ✓ | — | — | **Critical** |
| ✓ | ✗ | — | — | **High** |
| ✗ | — | ✓ | ✓ | **High** |
| ✗ | — | ✓ | ✗ | **Medium** |
| ✗ | — | ✗ | — | **Low** |

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Severity logic | `app/services/inference.py` | 430–461 |
| Unified video severity | `app/services/inference.py` | 540–549 |

---

## 6. ROI-based Seat Violation Detection

### Formula

**Center Point Check** — a person bounding box is inside the reserved seat ROI if its center falls within the ROI rectangle:

```
center_x = (box_x1 + box_x2) / 2
center_y = (box_y1 + box_y2) / 2

in_roi = (roi_x1 ≤ center_x ≤ roi_x2) AND (roi_y1 ≤ center_y ≤ roi_y2)
```

**ROI Coordinate Transformation** — ROI is stored as frame fractions [0.0, 1.0] and converted to pixel coordinates:

```
roi_x1_px = roi_x1_frac × frame_width
roi_y1_px = roi_y1_frac × frame_height
roi_x2_px = roi_x2_frac × frame_width
roi_y2_px = roi_y2_frac × frame_height
```

**Violation Condition**:

```
violation = (in_roi = True) AND (gender = "Male")
```

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| ROI center-point check | `app/services/inference.py` | 309–314 |
| ROI coordinate transform | `app/services/inference.py` | 333–336 |
| Violation detection | `app/services/inference.py` | 371–380 |
| ROI config (fractions) | `app/core/config.py` | 13–16 |

---

## 7. SHA-256 Hash Chain (Audit Trail)

### Formula

An append-only hash chain where each entry is cryptographically linked to its predecessor:

```
H₀ = "0000000000000000000000000000000000000000000000000000000000000000"
     (64-char zero string — genesis block)

Hₙ = SHA256( Hₙ₋₁ ‖ "|" ‖ record_type ‖ "|" ‖ JSON(payload) )
```

Where:
- `Hₙ` — hash of entry n (64-char hex string)
- `Hₙ₋₁` — hash of previous entry (backward link)
- `‖` — string concatenation
- `record_type` — event category (e.g., "violation", "alert")
- `JSON(payload)` — deterministic JSON serialization (sorted keys, no spaces)

### Tamper Detection Property

```
If any entry i is modified:
    H'ᵢ = SHA256(...modified...) ≠ Hᵢ (stored)
    → H'ᵢ₊₁ = SHA256(H'ᵢ ‖ ...) ≠ Hᵢ₊₁ (stored)
    → All subsequent hashes break → tampering detected
```

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| SHA-256 helper | `app/services/audit.py` | 12–13 |
| Genesis hash | `app/services/audit.py` | 16–26 |
| Chain append | `app/services/audit.py` | 29–41 |

---

## 8. Evidence Integrity Hashing

### Formula

Each piece of raw evidence (frame, video clip) is hashed and stored with a verifiable filename:

```
digest = SHA256(raw_bytes)

filename = "{UTC_timestamp}_{bus_id}_{digest[0:12]}.bin"
```

Where:
- `digest` — full 64-char SHA-256 hash of raw binary evidence
- `digest[0:12]` — first 12 characters used as filename suffix for uniqueness
- `UTC_timestamp` — format: `YYYYMMDDTHHMMSSz`

### Verification

```
verify(file) = ( SHA256(file.read_bytes()) == stored_digest )
```

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Hash function | `app/services/evidence.py` | 10–11 |
| Persist with naming | `app/services/evidence.py` | 14–20 |

---

## 9. YOLO Person Detection

### Formula

YOLO (You Only Look Once) performs single-pass object detection:

```
detections = YOLO(frame, conf ≥ θ, class = 0)

where:
    θ = 0.5           (confidence threshold, configurable)
    class 0 = "person" (COCO dataset class ID)

people_count = |detections|    (cardinality of detection set)
```

### Model Variants (YOLO26 Family)

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| yolo26n.pt | Nano | Fastest | Lower | Real-time edge |
| yolo26s.pt | Small | Fast | Moderate | Balanced edge |
| yolo26m.pt | Medium | Moderate | Good | **Default** |
| yolo26l.pt | Large | Slower | High | High-accuracy |
| yolo26x.pt | XLarge | Slowest | Highest | Maximum accuracy |

### Person Tracking (ByteTrack)

For temporally consistent detections across frames:

```
tracked_detections = YOLO.track(frame, conf ≥ θ, class = 0, tracker = "ByteTrack")
```

Each detection receives a persistent **track ID** that remains consistent across frames.

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Person counting | `app/services/inference.py` | 112–128 |
| Person tracking | `app/services/inference.py` | 130–159 |
| Model switching | `app/services/inference.py` | 103–110 |
| Available models | `app/services/inference.py` | 64–66 |
| Confidence threshold | `app/core/config.py` | 11 |

---

## 10. Password Hashing (Authentication)

### Formula

**PBKDF2-SHA256** (Password-Based Key Derivation Function 2):

```
hash = PBKDF2(password, salt, iterations, dkLen)

where:
    PRF = HMAC-SHA256
    salt = random(16 bytes)            (auto-generated)
    iterations = 600,000               (passlib default)
    dkLen = 32 bytes                   (derived key length)
```

**Verification**:

```
verify(input, stored) = ( PBKDF2(input, stored.salt, iterations) == stored.hash )
```

### JWT Token Generation

```
payload = { "sub": username, "exp": now_utc + 120 minutes }
token = HMAC_SHA256_sign(payload, secret_key)
```

### TOTP Two-Factor Authentication

```
TOTP(K, T) = HOTP(K, ⌊T / 30⌋)

where:
    K = base32_secret            (random 160-bit key)
    T = current_unix_timestamp
    30 = time step in seconds
    
HOTP(K, C) = Truncate(HMAC-SHA1(K, C)) mod 10⁶    (6-digit code)
```

### Source Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Password hash/verify | `app/core/security.py` | 11–19 |
| JWT creation | `app/core/security.py` | 22–25 |
| JWT decode | `app/core/security.py` | 28–38 |
| TOTP generation | `app/core/security.py` | 41–42 |
| TOTP verification | `app/core/security.py` | 45–47 |

---

## Summary — Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| MobileNetV2 forward pass | O(T × C × H × W) per clip | O(B × T × 1280) |
| Temporal mean pooling | O(T × 1280) | O(1280) |
| Frame differencing (fallback) | O(H × W × C) | O(H × W × C) |
| Trust score update | O(1) per scope | O(1) |
| Severity classification | O(1) | O(1) |
| ROI center-point check | O(1) per box | O(1) |
| SHA-256 hash chain append | O(n) for payload serialization | O(1) per entry |
| YOLO detection | O(H × W) per frame | O(K) detections |
| PBKDF2 key derivation | O(iterations) | O(dkLen) |
| TOTP verification | O(1) | O(1) |

---

*Generated for: Transport Policing System — Final Year Project*
*Date: February 2026*
