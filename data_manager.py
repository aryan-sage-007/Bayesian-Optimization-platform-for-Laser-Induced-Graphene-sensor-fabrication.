"""
data_manager.py — Persistent storage for LIG optimisation sessions
===================================================================
One JSON file per material, stored in  ./data/<material_id>.json

Data is written to disk after EVERY resistance entry, so the app
can be closed mid-session and resumed on a different day with
zero data loss.

JSON schema
-----------
{
    "material_id":    str,
    "name":           str,
    "carbon_content": float,
    "created_at":     ISO-8601 str,
    "batches": [
        {
            "batch_number": int,
            "type":         "lhs" | "bo",
            "created_at":   ISO-8601 str,
            "points": [
                {
                    "power":      float,
                    "speed":      int,
                    "resistance": float | null,   ← null = not measured yet
                    "timestamp":  ISO-8601 str | null
                },
                ...
            ]
        },
        ...
    ]
}
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")


# =============================================================================
# Internal helpers
# =============================================================================

def _ensure_dir():
    DATA_DIR.mkdir(exist_ok=True)


def _path(material_id: str) -> Path:
    return DATA_DIR / f"{material_id}.json"


def _write(material_id: str, data: dict):
    _ensure_dir()
    with open(_path(material_id), "w") as fh:
        json.dump(data, fh, indent=2)


# =============================================================================
# Material CRUD
# =============================================================================

def create_material(name: str, carbon_content: float) -> str:
    """Create a new material record on disk. Returns the new material_id."""
    _ensure_dir()
    mid = f"mat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    _write(mid, {
        "material_id":    mid,
        "name":           name.strip(),
        "carbon_content": float(carbon_content),
        "created_at":     datetime.now().isoformat(),
        "batches":        []
    })
    return mid


def load_material(material_id: str):
    """Load material data from disk. Returns None if the file does not exist."""
    p = _path(material_id)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def list_materials() -> list:
    """
    Return a summary list of all saved materials (newest first).
    Each entry is a dict with keys:
        material_id, name, carbon_content, created_at,
        n_complete, best_R, n_batches
    """
    _ensure_dir()
    results = []
    for p in sorted(DATA_DIR.glob("*.json"), reverse=True):
        try:
            with open(p) as fh:
                d = json.load(fh)
            done = get_all_experiments(d)
            results.append({
                "material_id":    d["material_id"],
                "name":           d["name"],
                "carbon_content": d["carbon_content"],
                "created_at":     d["created_at"][:10],
                "n_complete":     len(done),
                "best_R":         round(min(e["resistance"] for e in done), 3)
                                  if done else None,
                "n_batches":      len(d.get("batches", []))
            })
        except Exception:
            continue
    return results


# =============================================================================
# Batch management
# =============================================================================

def add_batch(material_id: str, batch_type: str, points: list) -> int:
    """
    Append a new batch (LHS or BO) to a material.

    Parameters
    ----------
    batch_type : "lhs" | "bo"
    points     : list of (power, speed) tuples — resistance starts as null

    Returns the new batch_number (1-indexed).
    """
    data = load_material(material_id)
    if data is None:
        raise ValueError(f"Material '{material_id}' not found.")
    num = len(data["batches"]) + 1
    data["batches"].append({
        "batch_number": num,
        "type":         batch_type,
        "created_at":   datetime.now().isoformat(),
        "points": [
            {"power": float(p), "speed": int(s),
             "resistance": None, "timestamp": None}
            for p, s in points
        ]
    })
    _write(material_id, data)
    return num


def update_resistance(material_id: str, batch_number: int,
                      point_index: int, resistance: float) -> bool:
    """
    Record a single resistance measurement and immediately save to disk.

    Returns True on success, False if the material or batch index is invalid.
    """
    data = load_material(material_id)
    if data is None:
        return False
    b_idx = batch_number - 1
    if b_idx < 0 or b_idx >= len(data["batches"]):
        return False
    pt               = data["batches"][b_idx]["points"][point_index]
    pt["resistance"] = round(float(resistance), 4)
    pt["timestamp"]  = datetime.now().isoformat()
    _write(material_id, data)
    return True


def get_current_batch(material_data: dict):
    """
    Find the most recent batch that still has unmeasured (null) points.

    Returns
    -------
    (batch_dict, list_of_missing_indices)  or  (None, None) if all done.
    """
    for batch in reversed(material_data.get("batches", [])):
        missing = [i for i, pt in enumerate(batch["points"])
                   if pt["resistance"] is None]
        if missing:
            return batch, missing
    return None, None


def get_all_experiments(material_data: dict) -> list:
    """
    Flatten all completed (resistance ≠ null) experiments across every batch.

    Each entry is a dict:
        batch, batch_type, power, speed, resistance, timestamp
    """
    out = []
    for b in material_data.get("batches", []):
        for pt in b["points"]:
            if pt["resistance"] is not None:
                out.append({
                    "batch":      b["batch_number"],
                    "batch_type": b["type"],
                    "power":      pt["power"],
                    "speed":      pt["speed"],
                    "resistance": pt["resistance"],
                    "timestamp":  pt.get("timestamp", "")
                })
    return out


# =============================================================================
# Transfer learning data
# =============================================================================

def get_transfer_data(current_id: str, current_carbon: float,
                      sigma: float = 20.0, min_sim: float = 0.05) -> list:
    """
    Collect experiments from every OTHER material, weighted by carbon-content
    similarity.

    Similarity formula
    ------------------
    w = exp( −|current_carbon − other_carbon| / sigma )

    With sigma = 20 (default):
        |Δcarbon|  →  similarity w
           0 %     →  1.00   (identical)
          10 %     →  0.61   (strong transfer)
          20 %     →  0.37   (moderate)
          40 %     →  0.14   (weak)
          60 %     →  0.05   (threshold — barely included)
          ≥ 70 %   →  < 0.03 (skipped)

    In the GP:  noise_level = base_noise + (1 − w) × 2
    so similar materials contribute almost as much as direct measurements,
    while dissimilar materials barely shift the posterior.

    Returns
    -------
    list of (power, speed, resistance, similarity_weight) tuples
    """
    transfer = []
    for mat in list_materials():
        if mat["material_id"] == current_id or mat["n_complete"] == 0:
            continue
        w = float(np.exp(-abs(current_carbon - mat["carbon_content"]) / sigma))
        if w < min_sim:
            continue
        d = load_material(mat["material_id"])
        if d is None:
            continue
        for e in get_all_experiments(d):
            transfer.append((e["power"], e["speed"], e["resistance"], w))
    return transfer
