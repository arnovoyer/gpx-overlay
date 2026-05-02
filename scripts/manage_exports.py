from pathlib import Path
from datetime import datetime
import json

p = Path.cwd() / "exports"
p.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
renamed = []
for f in p.iterdir():
    if not f.is_file():
        continue
    name = f.name
    # skip files that look already timestamped or are presets
    if name.startswith(ts) or name.startswith("preset-") or name.startswith("preset_"):
        continue
    new = p / f"{ts}-{name}"
    try:
        f.rename(new)
        renamed.append((name, new.name))
    except Exception as exc:
        print(f"ERROR renaming {name}: {exc}")

preset = {
    "overlay_preset": "Example",
    "overlay_theme": "dark",
    "unit_system": "metric",
    "card_field_1": "speed",
    "card_field_2": "dist",
    "card_field_3": "time",
    "minimap_window_km": 1.0,
    "chart_smooth_window": 7,
    "gauge_max_kph": 0,
    "show_gauge": True,
    "show_cards": True,
    "show_minimap": True,
    "show_elevation": False,
    "ui_mode": "Easy",
}

preset_name = f"preset-example-{ts}.json"
target = p / preset_name
with target.open("w", encoding="utf-8") as fh:
    json.dump(preset, fh, ensure_ascii=False, indent=2)

print("RENAMED:")
for a, b in renamed:
    print(f"{a} -> {b}")
print("CREATED:", target.name)
