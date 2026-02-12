# Deep Learning and Computer Vision Enabled Transport Policing System (MVP)

This repository contains a working MVP of a modular transport policing platform with:

- Real-time event ingestion API
- Violation detection and severity scoring
- Alert creation and acknowledgement
- Immutable-style evidence hashing
- Trust factor scoring for bus, route, and timeslot
- JWT auth + role checks + TOTP-based 2FA
- Minimal dashboard UI

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn app.main:app --reload
```

4. Open dashboard:

`http://127.0.0.1:8000/dashboard`
