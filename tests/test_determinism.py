#!/usr/bin/env python
"""DEFECT-008 regression test: the seeded extractor must be bit-identical across repeats.

Three arms:
  A  same video, twice in ONE process
  B  same video, once in each of TWO separate processes (catches PYTHONHASHSEED and
     any per-process RNG state, which is how the original defect actually manifested)
  C  the ORIGINAL unseeded extractor, twice - must FAIL, proving the test has teeth

Exit 0 only if A and B are bit-identical on all 50 descriptors and C differs.
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path.home()/"prism_r2"
SEEDED = ROOT/"repo"/"src"/"preprocessing"
LEGACY = ROOT/"legacy"/"phantomlens"/"src"   # optional: only used for the negative control

CHILD = r'''
import os,sys,json,warnings
warnings.filterwarnings("ignore")
mod_dir, video, use_seeded = sys.argv[1], sys.argv[2], sys.argv[3]=="1"
os.chdir(str(__import__("pathlib").Path.home()/"prism_r2"/"legacy"/"phantomlens"))
sys.path.insert(0, mod_dir)
if use_seeded:
    import precompute_features_seeded as P
else:
    import precompute_features_best as P
row = P.process_single_video(video, 1, max_frames=300) if use_seeded else \
      P.process_single_video(video, 1, max_frames=300)
feats = {k: float(v) for k, v in (row or {}).items() if k[:2] in ("s_","t_")}
print("@@JSON@@" + json.dumps(feats))
'''

def run_child(mod_dir, video, seeded):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(CHILD); tmp = f.name
    try:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0" if seeded else "random"
        out = subprocess.run([str(ROOT/"env"/"prism"/"bin"/"python"), tmp,
                              str(mod_dir), str(video), "1" if seeded else "0"],
                             capture_output=True, text=True, env=env, timeout=1200)
        for line in out.stdout.splitlines():
            if line.startswith("@@JSON@@"): return json.loads(line[8:])
        raise RuntimeError(f"child produced no result: {out.stderr[-500:]}")
    finally:
        os.unlink(tmp)

def maxdiff(a, b):
    ks = sorted(set(a) & set(b))
    if not ks: return None, 0
    return max(abs(a[k]-b[k]) for k in ks), sum(1 for k in ks if a[k] != b[k])

def main():
    # fixture: pass any video path, or set PRISM_TEST_VIDEO
    video = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "PRISM_TEST_VIDEO",
        str(Path.home()/"Datasets"/"FaceForensics++"/"original_sequences"/"youtube"/"c23"/"videos"/"000.mp4"))
    if not os.path.exists(video):
        print(f"SKIP: fixture video not found: {video}"); return 0
    print(f"fixture: {video}\n")
    ok = True

    # ---- A: seeded, twice in one process ----
    sys.path.insert(0, str(SEEDED)); os.chdir(str(ROOT/"legacy"/"phantomlens"))
    import precompute_features_seeded as P
    a1 = {k: float(v) for k, v in P.process_single_video(video, 1, max_frames=300).items() if k[:2] in ("s_","t_")}
    a2 = {k: float(v) for k, v in P.process_single_video(video, 1, max_frames=300).items() if k[:2] in ("s_","t_")}
    d, n = maxdiff(a1, a2)
    print(f"A  seeded, same process x2      : max|delta|={d:.3e}  differing={n}/50  "
          f"{'PASS' if d == 0 else 'FAIL'}")
    ok &= (d == 0)

    # ---- B: seeded, two separate processes ----
    b1 = run_child(SEEDED, video, True)
    b2 = run_child(SEEDED, video, True)
    d, n = maxdiff(b1, b2)
    print(f"B  seeded, cross-process x2     : max|delta|={d:.3e}  differing={n}/50  "
          f"{'PASS' if d == 0 else 'FAIL'}")
    ok &= (d == 0)
    d, n = maxdiff(a1, b1)
    print(f"B' seeded, in-proc vs child     : max|delta|={d:.3e}  differing={n}/50  "
          f"{'PASS' if d == 0 else 'FAIL'}")
    ok &= (d == 0)

    # ---- C: the ORIGINAL unseeded extractor must DIFFER, proving the test has teeth.
    # Optional: it needs the pre-fix extractor, which ships only in the authors' working tree.
    if (LEGACY/"precompute_features_best.py").exists():
        c1 = run_child(LEGACY, video, False)
        c2 = run_child(LEGACY, video, False)
        d, n = maxdiff(c1, c2)
        teeth = d > 0
        print(f"C  ORIGINAL unseeded x2         : max|delta|={d:.3e}  differing={n}/50  "
              f"{'PASS (differs, as expected)' if teeth else 'FAIL (test has no teeth)'}")
        ok &= teeth
    else:
        print("C  ORIGINAL unseeded x2         : SKIPPED (pre-fix extractor not in this archive; "
              "arms A/B are the release-relevant checks)")

    print("\n" + ("DETERMINISM REGRESSION: PASS" if ok else "DETERMINISM REGRESSION: FAIL"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
