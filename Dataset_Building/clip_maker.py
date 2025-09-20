from typing import List, Tuple, Union
from moviepy import VideoFileClip, concatenate_videoclips

# =============== USER SETTINGS ===============================================
INPUT_MP4  = ""
OUTPUT_MP4 = ""

# Time ranges to KEEP (seconds or "HH:MM:SS.sss").
# Example for a 30s clip starting at 123.4s: KEEP_RANGES = [(123.4, 153.4)]
KEEP_RANGES: List[Tuple[Union[float, str], Union[float, str]]] = [
    (11460, 11490)
]

OUTPUT_FPS = 1          # downsample to 1 fps
DROP_AUDIO = True       # set False if you want to keep audio
# ============================================================================

def _to_seconds(t: Union[str, float, int]) -> float:
    if isinstance(t, (int, float)): return float(t)
    parts = [float(p) for p in str(t).split(":")]
    if len(parts) == 3:
        h, m, s = parts; return h*3600 + m*60 + s
    if len(parts) == 2:
        m, s = parts; return m*60 + s
    return float(parts[0])

def _normalize_ranges(ranges, duration):
    out = []
    for a, b in ranges:
        s = max(0.0, min(_to_seconds(a), duration))
        e = max(0.0, min(_to_seconds(b), duration))
        if e > s: out.append((s, e))
    out.sort()
    merged = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged

def main():
    with VideoFileClip(INPUT_MP4) as clip:
        keep = _normalize_ranges(KEEP_RANGES, clip.duration)
        if not keep:
            raise ValueError("No valid KEEP_RANGES specified.")
        parts = [clip.subclipped(s, e) for s, e in keep]
        final = concatenate_videoclips(parts, method="compose")

        # Downsample to 1 fps (or whatever OUTPUT_FPS is)
        final = final.with_fps(OUTPUT_FPS)
        if DROP_AUDIO:
            final = final.without_audio()

        final.write_videofile(
            OUTPUT_MP4,
            codec="libx264",
            audio_codec=("aac" if not DROP_AUDIO else None),
            preset="medium",
            fps=OUTPUT_FPS,
            threads=0,
            temp_audiofile=None,
        )

if __name__ == "__main__":
    main()

