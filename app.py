from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import gpxpy
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "GPX Overlay Video"
DEFAULT_EMA_ALPHA = 0.22
DEFAULT_PREVIEW_GRAPH_SECONDS = 30.0
DEFAULT_MAX_SPEED_KPH = 120.0
DEFAULT_CARD_FIELDS = ("alt", "slope", "dist")
FONT_REGULAR = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_TRIPLEX
COLOR_PANEL = (18, 18, 22)
COLOR_PANEL_SOFT = (26, 26, 32)
COLOR_PANEL_BORDER = (64, 64, 72)
COLOR_TEXT = (245, 245, 245)
COLOR_MUTED = (176, 176, 176)
COLOR_ACCENT = (164, 232, 255)
COLOR_ACCENT_SOFT = (110, 180, 255)
COLOR_GOOD = (150, 235, 171)

CARD_FIELD_OPTIONS = {
    "alt": "ALT",
    "slope": "SLOPE",
    "dist": "DIST",
    "time": "TIME",
    "speed": "SPEED",
}


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    creation_time: Optional[float] = None
    creation_time_source: Optional[str] = None

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass(frozen=True)
class TrackData:
    time_s: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    elevation_m: np.ndarray
    distance_km: np.ndarray
    speed_kph_raw: np.ndarray
    speed_kph: np.ndarray
    slope_pct: np.ndarray
    start_time: float
    end_time: float

    @property
    def duration_s(self) -> float:
        return float(self.end_time - self.start_time)


@dataclass(frozen=True)
class TrackPoint:
    time_s: float
    lat: float
    lon: float
    elevation_m: float


def format_seconds(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def ensure_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def format_timestamp(timestamp_s: float) -> str:
    return datetime.fromtimestamp(timestamp_s, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def parse_creation_time(value: str) -> Optional[float]:
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def probe_video_creation_time(video_path: Path) -> tuple[Optional[float], Optional[str]]:
    if shutil.which("ffprobe") is None:
        return None, None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags=creation_time:stream_tags=creation_time",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return None, None

    candidates: list[tuple[str, str]] = []
    format_tags = payload.get("format", {}).get("tags", {})
    if isinstance(format_tags, dict):
        creation_time = format_tags.get("creation_time")
        if isinstance(creation_time, str):
            candidates.append((creation_time, "ffprobe.format.tags.creation_time"))

    for index, stream in enumerate(payload.get("streams", [])):
        if not isinstance(stream, dict):
            continue
        tags = stream.get("tags", {})
        if isinstance(tags, dict):
            creation_time = tags.get("creation_time")
            if isinstance(creation_time, str):
                candidates.append((creation_time, f"ffprobe.stream[{index}].tags.creation_time"))

    for raw_value, source in candidates:
        timestamp_s = parse_creation_time(raw_value)
        if timestamp_s is not None:
            return timestamp_s, source

    return None, None


def read_video_meta(video_path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Video konnte nicht geöffnet werden.")

    try:
        fps = safe_float(capture.get(cv2.CAP_PROP_FPS), 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0:
            raise ValueError("Ungültige FPS im Video.")
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Ungültige Videometadaten.")
        duration_s = frame_count / fps
        creation_time, creation_source = probe_video_creation_time(video_path)
        return VideoMeta(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
            creation_time=creation_time,
            creation_time_source=creation_source,
        )
    finally:
        capture.release()


def parse_gpx_points(gpx_path: Path) -> list[TrackPoint]:
    with gpx_path.open("r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)

    points: list[TrackPoint] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    continue
                time_s = point.time.timestamp()
                points.append(
                    TrackPoint(
                        time_s=time_s,
                        lat=float(point.latitude),
                        lon=float(point.longitude),
                        elevation_m=safe_float(point.elevation, 0.0),
                    )
                )

    if not points:
        raise ValueError("Keine GPX-Punkte mit Zeitstempeln gefunden.")

    points.sort(key=lambda entry: entry.time_s)
    filtered: list[TrackPoint] = [points[0]]
    for point in points[1:]:
        if point.time_s <= filtered[-1].time_s:
            continue
        filtered.append(point)

    if len(filtered) < 2:
        raise ValueError("GPX enthält zu wenige zeitlich getrennte Punkte.")

    return filtered


def smooth_ema(values: np.ndarray, alpha: float = DEFAULT_EMA_ALPHA) -> np.ndarray:
    if values.size == 0:
        return values
    alpha = float(np.clip(alpha, 0.01, 0.95))
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for idx in range(1, values.size):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def prepare_track(points: list[TrackPoint], alpha: float = DEFAULT_EMA_ALPHA) -> TrackData:
    time_abs = np.array([point.time_s for point in points], dtype=np.float64)
    lat = np.array([point.lat for point in points], dtype=np.float64)
    lon = np.array([point.lon for point in points], dtype=np.float64)
    elevation_m = np.array([point.elevation_m for point in points], dtype=np.float64)

    start_time = float(time_abs[0])
    time_s = time_abs - start_time

    segment_distances_m = np.zeros(time_s.shape[0], dtype=np.float64)
    segment_dt_s = np.zeros(time_s.shape[0], dtype=np.float64)

    for idx in range(1, time_s.shape[0]):
        segment_distances_m[idx] = haversine_m(lat[idx - 1], lon[idx - 1], lat[idx], lon[idx])
        segment_dt_s[idx] = max(1e-6, time_s[idx] - time_s[idx - 1])

    cumulative_distance_km = np.cumsum(segment_distances_m) / 1000.0
    raw_speed_kph = np.zeros(time_s.shape[0], dtype=np.float64)
    slope_pct = np.zeros(time_s.shape[0], dtype=np.float64)

    for idx in range(1, time_s.shape[0]):
        dt = segment_dt_s[idx]
        distance_m = segment_distances_m[idx]
        raw_speed_kph[idx] = (distance_m / dt) * 3.6 if dt > 0 else 0.0
        delta_height = elevation_m[idx] - elevation_m[idx - 1]
        slope_pct[idx] = (delta_height / distance_m) * 100.0 if distance_m > 1e-6 else 0.0

    speed_kph = smooth_ema(raw_speed_kph, alpha=alpha)
    end_time = float(time_abs[-1])
    return TrackData(
        time_s=time_s,
        lat=lat,
        lon=lon,
        elevation_m=elevation_m,
        distance_km=cumulative_distance_km,
        speed_kph_raw=raw_speed_kph,
        speed_kph=speed_kph,
        slope_pct=slope_pct,
        start_time=start_time,
        end_time=end_time,
    )


def derive_auto_offset(track: TrackData, video_meta: VideoMeta) -> Optional[float]:
    if video_meta.creation_time is None:
        return None
    return float(video_meta.creation_time - track.start_time)


def gpx_values_at_time(track: TrackData, video_time_s: float, offset_s: float) -> Optional[dict[str, float]]:
    relative_gpx_time = float(video_time_s + offset_s)
    if relative_gpx_time < track.time_s[0] or relative_gpx_time > track.time_s[-1]:
        return None

    idx = int(np.searchsorted(track.time_s, relative_gpx_time, side="left"))
    if idx <= 0:
        idx = 1
    if idx >= track.time_s.size:
        idx = track.time_s.size - 1

    t0 = track.time_s[idx - 1]
    t1 = track.time_s[idx]
    if t1 <= t0:
        return None

    ratio = (relative_gpx_time - t0) / (t1 - t0)

    def lerp(values: np.ndarray) -> float:
        return float(values[idx - 1] + ratio * (values[idx] - values[idx - 1]))

    return {
        "speed_kph": lerp(track.speed_kph),
        "elevation_m": lerp(track.elevation_m),
        "slope_pct": lerp(track.slope_pct),
        "distance_km": lerp(track.distance_km),
        "relative_gpx_time": relative_gpx_time,
    }


def interpolate_series(track: TrackData, values: np.ndarray, start_s: float, end_s: float, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 2 or end_s <= start_s:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    sample_times = np.linspace(start_s, end_s, num_points, dtype=np.float64)
    sample_values = np.interp(
        sample_times,
        track.time_s,
        values,
        left=np.nan,
        right=np.nan,
    )
    mask = np.isfinite(sample_values)
    return sample_times[mask], sample_values[mask]


def smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if window <= 1:
        return values
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.size]


def overlay_dimensions(width: int, height: int) -> dict[str, int]:
    scale = max(0.75, min(width, height) / 900.0)
    return {
        "pad": max(14, int(20 * scale)),
        "label_scale": max(0.48, 0.62 * scale),
        "small_scale": max(0.42, 0.50 * scale),
        "big_scale": max(1.0, 1.75 * scale),
        "label_thickness": max(1, int(round(1.0 * scale))),
        "big_thickness": max(2, int(round(3.0 * scale))),
        "graph_height": max(60, int(0.12 * height)),
        "graph_stroke": max(2, int(round(2.0 * scale))),
        "graph_fill_alpha": 0.34,
        "card_radius": max(16, int(22 * scale)),
        "card_border": max(1, int(round(1.5 * scale))),
        "gauge_radius": max(42, int(min(width, height) * 0.11)),
    }


def resolve_metric_card(field_key: str, metrics: dict[str, float]) -> tuple[str, str, tuple[int, int, int]]:
    key = field_key.lower().strip()
    if key == "alt":
        return "ALT", f"{metrics['elevation_m']:.0f} m", COLOR_GOOD
    if key == "slope":
        return "SLOPE", f"{metrics['slope_pct']:.1f} %", COLOR_ACCENT_SOFT
    if key == "dist":
        return "DIST", f"{metrics['distance_km']:.2f} km", COLOR_ACCENT
    if key == "time":
        return "TIME", format_seconds(metrics["relative_gpx_time"]), COLOR_MUTED
    if key == "speed":
        return "SPEED", f"{max(0.0, metrics['speed_kph']):.1f} kph", COLOR_ACCENT
    return "ALT", f"{metrics['elevation_m']:.0f} m", COLOR_GOOD


def draw_text_with_shadow(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    font: int = FONT_REGULAR,
) -> None:
    shadow_color = (0, 0, 0)
    x, y = pos
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, shadow_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(frame: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], radius: int, thickness: int = -1) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        return

    if thickness < 0:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def add_layer(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    blended = base.copy()
    cv2.addWeighted(layer, alpha, blended, 1.0, 0.0, blended)
    return blended


def draw_speed_gauge(layer: np.ndarray, dims: dict[str, int], box: tuple[int, int, int, int], speed_kph: float, speed_cap: float) -> None:
    x, y, w, h = box
    center = (x + int(w * 0.50), y + int(h * 0.70))
    radius = min(int(w * 0.34), int(h * 0.34), dims["gauge_radius"])
    start_angle = 150
    sweep = 240
    end_angle = start_angle + sweep

    cv2.ellipse(layer, center, (radius, radius), 0, start_angle, end_angle, COLOR_PANEL_BORDER, dims["card_border"] + 1, cv2.LINE_AA)
    ratio = clamp(speed_kph / max(1.0, speed_cap), 0.0, 1.0)
    accent_angle = start_angle + sweep * ratio
    cv2.ellipse(layer, center, (radius, radius), 0, start_angle, int(accent_angle), COLOR_ACCENT, dims["card_border"] + 2, cv2.LINE_AA)

    for tick in range(0, 11):
        tick_ratio = tick / 10.0
        tick_angle = math.radians((start_angle + sweep * tick_ratio) - 90.0)
        outer = int(radius * 1.03)
        inner = int(radius * (0.86 if tick % 2 == 0 else 0.90))
        x1 = int(center[0] + inner * math.cos(tick_angle))
        y1 = int(center[1] + inner * math.sin(tick_angle))
        x2 = int(center[0] + outer * math.cos(tick_angle))
        y2 = int(center[1] + outer * math.sin(tick_angle))
        cv2.line(layer, (x1, y1), (x2, y2), (110, 110, 116), 1, cv2.LINE_AA)

    needle_angle = math.radians(accent_angle - 90.0)
    needle_length = int(radius * 0.92)
    needle_x = int(center[0] + needle_length * math.cos(needle_angle))
    needle_y = int(center[1] + needle_length * math.sin(needle_angle))
    cv2.line(layer, center, (needle_x, needle_y), (20, 20, 20), max(4, dims["card_border"] + 2), cv2.LINE_AA)
    cv2.line(layer, center, (needle_x, needle_y), COLOR_TEXT, max(2, dims["card_border"] + 1), cv2.LINE_AA)
    cv2.circle(layer, center, max(5, dims["card_border"] + 3), COLOR_TEXT, -1, cv2.LINE_AA)
    cv2.circle(layer, center, max(2, dims["card_border"] + 1), COLOR_ACCENT_SOFT, -1, cv2.LINE_AA)

    value_text = f"{speed_kph:0.1f}"
    text_scale = dims["big_scale"] * 0.88
    text_size, _ = cv2.getTextSize(value_text, FONT_BOLD, text_scale, dims["big_thickness"])
    text_x = x + int((w - text_size[0]) * 0.5)
    text_y = y + int(h * 0.24)
    draw_text_with_shadow(layer, value_text, (text_x, text_y), text_scale, COLOR_TEXT, dims["big_thickness"], font=FONT_BOLD)
    label = "KPH"
    label_size, _ = cv2.getTextSize(label, FONT_REGULAR, dims["label_scale"], dims["label_thickness"])
    draw_text_with_shadow(layer, label, (x + int((w - label_size[0]) * 0.5), y + int(h * 0.36)), dims["label_scale"], COLOR_MUTED, dims["label_thickness"], font=FONT_REGULAR)


def draw_metric_card(layer: np.ndarray, dims: dict[str, int], box: tuple[int, int, int, int], label: str, value: str, accent: tuple[int, int, int]) -> None:
    x, y, w, h = box
    inner_pad = max(8, int(10 * min(w, h) / 90))
    draw_rounded_rect(layer, (x, y), (x + w, y + h), COLOR_PANEL, dims["card_radius"])
    draw_rounded_rect(layer, (x, y), (x + w, y + h), COLOR_PANEL_BORDER, dims["card_radius"], thickness=dims["card_border"])
    cv2.line(layer, (x + inner_pad, y + inner_pad + 1), (x + inner_pad + 20, y + inner_pad + 1), accent, max(2, dims["card_border"]), cv2.LINE_AA)
    label_scale = dims["small_scale"] * 0.88
    label_y = y + inner_pad + 14
    draw_text_with_shadow(layer, label, (x + inner_pad, label_y), label_scale, COLOR_MUTED, dims["label_thickness"], font=FONT_REGULAR)
    (_, label_h), _ = cv2.getTextSize(label, FONT_REGULAR, label_scale, dims["label_thickness"])
    value_scale = dims["label_scale"] * 1.16
    value_y = min(y + h - inner_pad - 4, label_y + label_h + max(10, int(h * 0.22)))
    draw_text_with_shadow(layer, value, (x + inner_pad, value_y), value_scale, COLOR_TEXT, dims["big_thickness"], font=FONT_BOLD)


def draw_profile_chart(
    layer: np.ndarray,
    dims: dict[str, int],
    box: tuple[int, int, int, int],
    track: TrackData,
    current_time: float,
    smooth_window: int = 5,
    title: str = "Elevation",
) -> None:
    x, y, w, h = box
    chart_layer = np.zeros_like(layer)
    draw_rounded_rect(chart_layer, (x, y), (x + w, y + h), COLOR_PANEL_SOFT, dims["card_radius"])
    draw_rounded_rect(chart_layer, (x, y), (x + w, y + h), COLOR_PANEL_BORDER, dims["card_radius"], thickness=dims["card_border"])

    content_x1 = x + dims["pad"]
    content_y1 = y + dims["pad"]
    content_x2 = x + w - dims["pad"]
    content_y2 = y + h - dims["pad"]
    sample_start = max(0.0, current_time - DEFAULT_PREVIEW_GRAPH_SECONDS)
    sample_end = min(track.time_s[-1], max(current_time, sample_start + 1.0))
    sample_times, sample_values = interpolate_series(track, track.elevation_m, sample_start, sample_end, max(80, int(w / 8)))
    sample_values = smooth_signal(sample_values, smooth_window)

    draw_text_with_shadow(chart_layer, title.upper(), (content_x1, content_y1 + 18), dims["small_scale"], COLOR_MUTED, dims["label_thickness"], font=FONT_REGULAR)

    if sample_values.size >= 2:
        min_value = float(np.nanmin(sample_values))
        max_value = float(np.nanmax(sample_values))
        if math.isclose(min_value, max_value):
            max_value = min_value + 1.0

        usable_h = max(20, content_y2 - content_y1 - 18)
        usable_w = max(20, content_x2 - content_x1)
        baseline_y = content_y2

        points = []
        fill_points = [[int(content_x1), baseline_y]]
        for sample_time, sample_value in zip(sample_times, sample_values):
            ratio_x = 0.0 if sample_end <= sample_start else (sample_time - sample_start) / (sample_end - sample_start)
            x_pos = int(content_x1 + ratio_x * usable_w)
            normalized = (sample_value - min_value) / (max_value - min_value)
            y_pos = int(content_y2 - normalized * usable_h)
            points.append([x_pos, y_pos])
            fill_points.append([x_pos, y_pos])
        fill_points.append([int(content_x2), baseline_y])

        if len(fill_points) >= 3:
            cv2.fillPoly(chart_layer, [np.array(fill_points, dtype=np.int32)], (28, 36, 44))
        if len(points) >= 2:
            cv2.polylines(chart_layer, [np.array(points, dtype=np.int32)], False, COLOR_ACCENT_SOFT, dims["graph_stroke"], cv2.LINE_AA)

        if sample_start <= current_time <= sample_end:
            ratio_x = 0.0 if sample_end <= sample_start else (current_time - sample_start) / (sample_end - sample_start)
            current_x = int(content_x1 + ratio_x * usable_w)
            current_value = float(np.interp(current_time, track.time_s, track.elevation_m, left=np.nan, right=np.nan))
            if math.isfinite(current_value):
                normalized = (current_value - min_value) / (max_value - min_value)
                current_y = int(content_y2 - normalized * usable_h)
                cv2.line(chart_layer, (current_x, content_y1 + 8), (current_x, baseline_y), (92, 92, 100), 1, cv2.LINE_AA)
                cv2.circle(chart_layer, (current_x, current_y), max(4, dims["graph_stroke"] + 1), COLOR_TEXT, -1, cv2.LINE_AA)

    cv2.addWeighted(chart_layer, 0.78, layer, 1.0, 0.0, layer)


def draw_overlay(
    frame: np.ndarray,
    metrics: Optional[dict[str, float]],
    track: TrackData,
    video_time_s: float,
    offset_s: float,
    card_fields: tuple[str, str, str],
    chart_smooth_window: int,
    gauge_max_kph: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    dims = overlay_dimensions(width, height)
    pad = dims["pad"]
    overlay = frame.copy()
    background = np.zeros_like(frame)

    speed_box = (pad, pad, int(width * 0.28), int(height * 0.21))
    card_w = int(width * 0.155)
    card_h = int(height * 0.066)
    card_x = width - pad - card_w
    card_y_top = pad
    card_gap = max(6, int(pad * 0.42))

    graph_h = dims["graph_height"]
    graph_box = (pad, height - graph_h - pad, width - 2 * pad, graph_h)

    if metrics is not None:
        speed_kph = max(0.0, metrics["speed_kph"])
        auto_cap = max(28.0, float(np.nanpercentile(track.speed_kph, 90)) * 1.10)
        cap = gauge_max_kph if gauge_max_kph > 0 else auto_cap
        draw_rounded_rect(background, (speed_box[0], speed_box[1]), (speed_box[0] + speed_box[2], speed_box[1] + speed_box[3]), COLOR_PANEL, dims["card_radius"])
        draw_rounded_rect(background, (speed_box[0], speed_box[1]), (speed_box[0] + speed_box[2], speed_box[1] + speed_box[3]), COLOR_PANEL_BORDER, dims["card_radius"], thickness=dims["card_border"])
        draw_speed_gauge(background, dims, speed_box, speed_kph, cap)

        for idx, field in enumerate(card_fields):
            label, value, accent = resolve_metric_card(field, metrics)
            box_y = card_y_top + idx * (card_h + card_gap)
            draw_metric_card(background, dims, (card_x, box_y, card_w, card_h), label, value, accent)

    draw_profile_chart(background, dims, graph_box, track, video_time_s + offset_s, smooth_window=chart_smooth_window, title="Elevation")

    overlay = add_layer(overlay, background, 0.92)

    if metrics is None:
        warning = "GPX-Daten für diesen Videobereich nicht verfügbar"
        box_w = int(width * 0.66)
        box_h = max(64, int(64 * dims["label_scale"] * 1.4))
        box_x = pad
        box_y = height // 2 - box_h // 2
        warn_layer = np.zeros_like(frame)
        draw_rounded_rect(warn_layer, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_PANEL, dims["card_radius"])
        draw_rounded_rect(warn_layer, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_PANEL_BORDER, dims["card_radius"], thickness=dims["card_border"])
        draw_text_with_shadow(warn_layer, warning, (box_x + pad, box_y + box_h // 2 + 9), dims["label_scale"], COLOR_TEXT, dims["label_thickness"], font=FONT_BOLD)
        overlay = add_layer(overlay, warn_layer, 0.95)

    return overlay


def read_frame_at(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        success, frame = capture.read()
        if not success:
            return None
        return frame
    finally:
        capture.release()


def export_video(
    video_meta: VideoMeta,
    track: TrackData,
    offset_s: float,
    output_path: Path,
    progress_placeholder: Optional[st.delta_generator.DeltaGenerator] = None,
    card_fields: tuple[str, str, str] = DEFAULT_CARD_FIELDS,
    chart_smooth_window: int = 7,
    gauge_max_kph: float = 0.0,
) -> Path:
    if not ensure_ffmpeg_available():
        raise RuntimeError("FFmpeg wurde nicht gefunden. Bitte FFmpeg installieren oder in PATH verfügbar machen.")

    temp_dir = Path(tempfile.mkdtemp(prefix="gpx_overlay_"))
    temp_video_path = temp_dir / "video_no_audio.mp4"
    audio_path = temp_dir / "audio.m4a"

    writer = cv2.VideoWriter(
        str(temp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_meta.fps,
        (video_meta.width, video_meta.height),
    )
    if not writer.isOpened():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("OpenCV VideoWriter konnte nicht geöffnet werden.")

    capture = cv2.VideoCapture(str(video_meta.path))
    if not capture.isOpened():
        writer.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Video konnte für den Export nicht geöffnet werden.")

    try:
        for frame_idx in range(video_meta.frame_count):
            success, frame = capture.read()
            if not success:
                break
            video_time_s = frame_idx / video_meta.fps
            metrics = gpx_values_at_time(track, video_time_s, offset_s)
            rendered = draw_overlay(frame, metrics, track, video_time_s, offset_s, card_fields, chart_smooth_window, gauge_max_kph)
            writer.write(rendered)
            if progress_placeholder is not None:
                progress_placeholder.progress(min(1.0, (frame_idx + 1) / video_meta.frame_count))
    finally:
        capture.release()
        writer.release()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video_path),
        "-i",
        str(video_meta.path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return output_path


def save_upload(uploaded_file) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="gpx_overlay_upload_"))
    suffix = Path(uploaded_file.name).suffix.lower() or ".bin"
    target = temp_dir / f"upload{suffix}"
    with target.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target


@st.cache_data(show_spinner=False)
def cached_video_meta(video_path: str) -> VideoMeta:
    return read_video_meta(Path(video_path))


@st.cache_data(show_spinner=False)
def cached_track_data(gpx_path: str, alpha: float) -> TrackData:
    points = parse_gpx_points(Path(gpx_path))
    return prepare_track(points, alpha=alpha)


def reset_session_state() -> None:
    defaults = {
        "video_path": None,
        "gpx_path": None,
        "output_path": None,
        "video_meta": None,
        "manual_offset_s": 0.0,
        "fine_tune_s": 0.0,
        "use_auto_sync": True,
        "chart_smooth_window": 7,
        "card_field_1": DEFAULT_CARD_FIELDS[0],
        "card_field_2": DEFAULT_CARD_FIELDS[1],
        "card_field_3": DEFAULT_CARD_FIELDS[2],
        "gauge_max_kph": 0,
        "export_directory": str((Path.cwd() / "exports").resolve()),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_files(video_file, gpx_file) -> None:
    if video_file is not None:
        st.session_state.video_path = save_upload(video_file)
    if gpx_file is not None:
        st.session_state.gpx_path = save_upload(gpx_file)


def preview_frame(
    video_meta: VideoMeta,
    track: TrackData,
    offset_s: float,
    frame_index: int,
    card_fields: tuple[str, str, str],
    chart_smooth_window: int,
    gauge_max_kph: float,
) -> None:
    frame = read_frame_at(video_meta.path, frame_index)
    if frame is None:
        st.error("Frame konnte nicht geladen werden.")
        return

    video_time_s = frame_index / video_meta.fps
    metrics = gpx_values_at_time(track, video_time_s, offset_s)
    rendered = draw_overlay(frame, metrics, track, video_time_s, offset_s, card_fields, chart_smooth_window, gauge_max_kph)
    rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
    st.image(rendered_rgb, use_container_width=True, caption=f"Frame {frame_index} bei {format_seconds(video_time_s)}")


def render_header(video_meta: VideoMeta, track: TrackData) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Video", f"{video_meta.width} x {video_meta.height}")
    col2.metric("FPS", f"{video_meta.fps:.2f}")
    col3.metric("Dauer", format_seconds(video_meta.duration_s))
    col4.metric("GPX-Punkte", f"{track.time_s.size}")


def export_ui(
    video_meta: VideoMeta,
    track: TrackData,
    offset_s: float,
    card_fields: tuple[str, str, str],
    chart_smooth_window: int,
    gauge_max_kph: float,
) -> None:
    output_dir = st.text_input("Export-Ordner", value=st.session_state.export_directory, key="export_directory")
    output_name = st.text_input("Export-Dateiname", value="overlay_output.mp4")
    export_clicked = st.button("Video mit Overlay exportieren", type="primary")
    if not export_clicked:
        return

    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / output_name
    progress = st.progress(0.0)
    status = st.empty()
    try:
        status.info("Export läuft ...")
        result = export_video(
            video_meta,
            track,
            offset_s,
            output_path,
            progress_placeholder=progress,
            card_fields=card_fields,
            chart_smooth_window=chart_smooth_window,
            gauge_max_kph=gauge_max_kph,
        )
        status.success(f"Export abgeschlossen: {result}")
        with result.open("rb") as handle:
            st.download_button("MP4 herunterladen", data=handle, file_name=result.name, mime="video/mp4")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        status.error(f"FFmpeg-Fehler beim Export: {stderr or exc}")
    except Exception as exc:
        status.error(str(exc))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    reset_session_state()

    st.title("GPX Overlay Video")
    st.caption("Video und GPX werden synchronisiert, direkt auf Frames gezeichnet und als MP4 exportiert.")

    with st.sidebar:
        st.header("Dateien")
        video_file = st.file_uploader("Video", type=["mp4", "mov", "mkv", "avi", "m4v"])
        gpx_file = st.file_uploader("GPX", type=["gpx"])
        ema_alpha = st.slider("Geschwindigkeitsglättung", min_value=0.05, max_value=0.60, value=DEFAULT_EMA_ALPHA, step=0.01)
        if st.button("Dateien laden"):
            try:
                load_files(video_file, gpx_file)
                st.success("Dateien geladen.")
            except Exception as exc:
                st.error(str(exc))

    if not st.session_state.video_path or not st.session_state.gpx_path:
        st.info("Bitte Video und GPX hochladen und dann 'Dateien laden' klicken.")
        return

    try:
        video_meta: VideoMeta = st.session_state.video_meta or cached_video_meta(str(st.session_state.video_path))
        track_data: TrackData = cached_track_data(str(st.session_state.gpx_path), ema_alpha)
        st.session_state.video_meta = video_meta
    except Exception as exc:
        st.error(str(exc))
        return

    if not ensure_ffmpeg_available():
        st.warning("FFmpeg wurde nicht gefunden. Export wird ohne funktionierende Encode-Kette nicht möglich sein.")

    render_header(video_meta, track_data)

    auto_offset_s = derive_auto_offset(track_data, video_meta)
    with st.sidebar:
        st.header("Synchronisation")
        auto_sync_enabled = st.checkbox("Auto-Sync per Video-Timestamp", value=st.session_state.use_auto_sync, key="use_auto_sync")
        if auto_offset_s is not None:
            st.metric("Auto-Offset", f"{auto_offset_s:+.1f} s")
            st.caption(f"Video-Start: {format_timestamp(video_meta.creation_time or 0.0)}")
            if video_meta.creation_time_source:
                st.caption(f"Quelle: {video_meta.creation_time_source}")
        else:
            st.warning("Kein verwertbarer Videozeitstempel gefunden. Manueller Offset wird verwendet.")

        manual_offset_s = st.slider(
            "Manueller GPX-Offset in Sekunden",
            min_value=-600.0,
            max_value=600.0,
            value=float(st.session_state.manual_offset_s),
            step=0.1,
            key="manual_offset_s",
        )
        fine_tune_s = st.slider(
            "Feinabgleich bei Auto-Sync",
            min_value=-30.0,
            max_value=30.0,
            value=float(st.session_state.fine_tune_s),
            step=0.1,
            key="fine_tune_s",
        )

        st.header("Overlay Layout")
        card_field_1 = st.selectbox(
            "Karte 1",
            options=list(CARD_FIELD_OPTIONS.keys()),
            format_func=lambda key: CARD_FIELD_OPTIONS[key],
            key="card_field_1",
        )
        card_field_2 = st.selectbox(
            "Karte 2",
            options=list(CARD_FIELD_OPTIONS.keys()),
            format_func=lambda key: CARD_FIELD_OPTIONS[key],
            key="card_field_2",
        )
        card_field_3 = st.selectbox(
            "Karte 3",
            options=list(CARD_FIELD_OPTIONS.keys()),
            format_func=lambda key: CARD_FIELD_OPTIONS[key],
            key="card_field_3",
        )
        chart_smooth_window = st.slider(
            "Elevation Smooth",
            min_value=1,
            max_value=21,
            value=int(st.session_state.chart_smooth_window),
            step=2,
            key="chart_smooth_window",
        )
        gauge_max_kph = st.slider(
            "Gauge Max KPH (0 = Auto)",
            min_value=0,
            max_value=180,
            value=int(st.session_state.gauge_max_kph),
            step=5,
            key="gauge_max_kph",
        )

    card_fields = (card_field_1, card_field_2, card_field_3)

    if auto_sync_enabled and auto_offset_s is not None:
        offset_s = auto_offset_s + fine_tune_s
    else:
        offset_s = manual_offset_s

    st.subheader("Timesync und Preview")
    col1, col2 = st.columns([2, 1])
    with col1:
        frame_index = st.slider("Preview-Frame", min_value=0, max_value=max(0, video_meta.frame_count - 1), value=0, step=1)
    with col2:
        current_video_time = frame_index / video_meta.fps
        st.metric("Videotime", format_seconds(current_video_time))
        current_gpx_time = current_video_time + offset_s
        st.metric("Effektiver Offset", f"{offset_s:+.1f} s")
        st.metric("GPX-Zeit relativ", format_seconds(max(0.0, current_gpx_time)))
        if current_gpx_time < 0:
            st.warning("Der Offset verschiebt den GPX-Start vor den Videostart.")
        elif current_gpx_time > track_data.time_s[-1]:
            st.warning("Der aktuelle Videopunkt liegt hinter dem Ende der GPX-Daten.")

    preview_frame(video_meta, track_data, offset_s, frame_index, card_fields, chart_smooth_window, float(gauge_max_kph))

    st.subheader("Export")
    export_ui(video_meta, track_data, offset_s, card_fields, chart_smooth_window, float(gauge_max_kph))

    with st.expander("GPX-Statistik", expanded=False):
        stats = pd.DataFrame(
            {
                "min": [float(np.nanmin(track_data.speed_kph)), float(np.nanmin(track_data.elevation_m)), float(np.nanmin(track_data.slope_pct)), float(np.nanmin(track_data.distance_km))],
                "max": [float(np.nanmax(track_data.speed_kph)), float(np.nanmax(track_data.elevation_m)), float(np.nanmax(track_data.slope_pct)), float(np.nanmax(track_data.distance_km))],
                "mean": [float(np.nanmean(track_data.speed_kph)), float(np.nanmean(track_data.elevation_m)), float(np.nanmean(track_data.slope_pct)), float(np.nanmean(track_data.distance_km))],
            },
            index=["speed_kph", "elevation_m", "slope_pct", "distance_km"],
        )
        st.dataframe(stats)


if __name__ == "__main__":
    main()