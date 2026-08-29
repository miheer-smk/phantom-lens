"""PRISM-50 operating-domain check (Phase 1.7b).

PRISM-50 is defined for full-frame video at native frame rate with intact acquisition and
encoding history. Datasets distributed as pre-cropped face sequences fall outside that domain:
descriptors that depend on an absent substrate do not fail loudly, they emit plausible but
meaningless values (see DEFECT-003), while length-gated descriptors are silently zero-filled
(DEFECT-006).

`check_substrate()` makes those preconditions machine-checkable BEFORE evaluation, so a caller
learns that a dataset is out of domain rather than discovering it from an uninterpretable AUC.

Ships in the Zenodo release as a usability feature, not only as an audit artifact.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# 1.7a - the four substrate preconditions and the descriptors that depend on each
# ---------------------------------------------------------------------------

BACKGROUND_DEPENDENT = [
    "s_prnu_face_periph",        # face-vs-periphery residual energy ratio
    "s_face_bg_diff",            # face-to-background luminance difference
    "t_prnu_face_vs_bg",         # PRNU face-vs-background stability
    "t_boundary_grad_temporal",  # boundary coherence (3)
    "t_boundary_color_disc",
    "t_boundary_freq_leakage",
    "t_skin_bg_decorrelation",   # skin-background decorrelation
]

CODEC_DEPENDENT = [
    "s_benford_dev",             # Benford deviation of DCT coefficients
    "s_block_artifact",          # 8x8 blocking energy
    "s_dbl_compress",            # double-compression signature
    "t_residual_flow_corr",      # codec temporal residual (2)
    "t_residual_entropy",
]

FRAMERATE_DEPENDENT = [
    "t_blink_rate",              # events per second - needs true fps
    "t_blink_duration",          # seconds - needs true fps
    "t_rppg_snr",                # rPPG band-pass is defined in Hz
    "t_rppg_peak_prominence",
    "t_rppg_interregion_corr",
    "t_rppg_harmonic_ratio",
]

RPPG_GATED = ["t_rppg_snr", "t_rppg_peak_prominence",
              "t_rppg_interregion_corr", "t_rppg_harmonic_ratio"]

MIN_FRAMES_SPATIAL, MIN_FRAMES_TEMPORAL, MIN_FRAMES_RPPG = 10, 30, 60

# the 33 temporal descriptors gated at 30 frames = all t_* except the 4 rPPG terms
TEMPORAL_GATED_COUNT = 33


@dataclass
class SubstrateReport:
    source: str
    passes: dict = field(default_factory=dict)
    n_undefined: int = 0          # descriptors with no valid substrate at all
    n_unreliable: int = 0         # descriptors computed on a degenerate substrate
    undefined: list = field(default_factory=list)
    unreliable: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def in_operating_domain(self) -> bool:
        return all(self.passes.values())

    def to_dict(self):
        d = asdict(self); d["in_operating_domain"] = self.in_operating_domain; return d


def check_substrate(source,
                    n_frames: int | None = None,
                    native_fps: float | None = None,
                    is_cropped_face: bool | None = None,
                    container: str | None = None) -> SubstrateReport:
    """Check the four PRISM-50 substrate preconditions.

    source            path to a video, or a directory / sequence of image frames
    n_frames          frames available after sampling (measured if omitted and readable)
    native_fps        true capture frame rate; None means unknown -> precondition fails
    is_cropped_face   True if frames are pre-cropped faces with no background context
    container         'video' | 'image_sequence'; inferred when omitted

    Returns a SubstrateReport. It never raises on an unreadable source; it reports.
    """
    src = Path(source) if not isinstance(source, (list, tuple)) else Path(str(source[0]).rsplit("/", 1)[0])
    if container is None:
        container = "image_sequence" if (src.is_dir() or (isinstance(source, (list, tuple)))) else "video"

    # --- measure what was not supplied -------------------------------------------------
    if n_frames is None or (native_fps is None and container == "video"):
        try:
            import cv2
            if container == "video":
                cap = cv2.VideoCapture(str(src))
                if cap.isOpened():
                    if n_frames is None:
                        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
                    if native_fps is None:
                        f = cap.get(cv2.CAP_PROP_FPS)
                        native_fps = float(f) if f and f > 0 else None
                    cap.release()
            elif n_frames is None:
                n_frames = len(list(src.glob("*.png")) + list(src.glob("*.jpg")))
        except Exception:
            pass

    if is_cropped_face is None:
        # an image-sequence distribution of face crops is the case this guards against;
        # callers with ground truth should pass the flag explicitly
        is_cropped_face = (container == "image_sequence")

    nf = n_frames or 0

    # --- 1.7a preconditions -------------------------------------------------------------
    p_background = not is_cropped_face
    p_codec      = (container == "video")     # PNG/JPG frame exports discard the original bitstream
    p_framerate  = native_fps is not None and native_fps > 0 and container == "video"
    p_length     = nf >= MIN_FRAMES_RPPG

    rep = SubstrateReport(source=str(source))
    rep.passes = {"background_context_present": bool(p_background),
                  "original_codec_intact":      bool(p_codec),
                  "native_frame_rate_known":    bool(p_framerate),
                  "sequence_length_sufficient": bool(p_length)}

    # --- descriptor accounting ----------------------------------------------------------
    unreliable, undefined = set(), set()
    if not p_background:
        unreliable |= set(BACKGROUND_DEPENDENT)   # computed on a crop rim -> plausible but wrong
    if not p_codec:
        unreliable |= set(CODEC_DEPENDENT)        # computed on a re-encode -> plausible but wrong
    if not p_framerate:
        unreliable |= set(FRAMERATE_DEPENDENT)    # assumed fps -> scaled wrongly
    if nf < MIN_FRAMES_RPPG:
        undefined |= set(RPPG_GATED)              # zero-filled
    if nf < MIN_FRAMES_TEMPORAL:
        undefined |= {"<all 33 temporal descriptors gated at 30 frames>"}
    if nf < MIN_FRAMES_SPATIAL:
        undefined |= {"<video excluded entirely: below MIN_FRAMES_SPATIAL>"}

    unreliable -= {u for u in undefined if not u.startswith("<")}   # undefined dominates
    rep.undefined  = sorted(undefined)
    rep.unreliable = sorted(unreliable)
    rep.n_undefined = (len(RPPG_GATED) if nf < MIN_FRAMES_RPPG else 0) + \
                      (TEMPORAL_GATED_COUNT if nf < MIN_FRAMES_TEMPORAL else 0)
    rep.n_unreliable = len(rep.unreliable)
    rep.detail = {"container": container, "n_frames": nf, "native_fps": native_fps,
                  "is_cropped_face": bool(is_cropped_face),
                  "gates": {"spatial": MIN_FRAMES_SPATIAL, "temporal": MIN_FRAMES_TEMPORAL,
                            "rppg": MIN_FRAMES_RPPG}}
    return rep


def summarise(reports: Sequence[SubstrateReport]) -> dict:
    """Dataset-level roll-up over per-video reports."""
    n = len(reports) or 1
    keys = ["background_context_present", "original_codec_intact",
            "native_frame_rate_known", "sequence_length_sufficient"]
    return {"n_videos": len(reports),
            "pass_rate": {k: round(sum(r.passes.get(k, False) for r in reports) / n, 4) for k in keys},
            "fully_in_domain_rate": round(sum(r.in_operating_domain for r in reports) / n, 4),
            "mean_n_undefined":  round(sum(r.n_undefined for r in reports) / n, 2),
            "mean_n_unreliable": round(sum(r.n_unreliable for r in reports) / n, 2)}
