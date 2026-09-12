"""
Reorganizes freshly-downloaded PSD / SHIQ / SSHR archives into the folder
layout that UnReflectAnything_Dataset (see unreflectdataset.py + wrappers.py)
expects under a single $DATASET_DIR root:

    $DATASET_DIR/PSD_Dataset/{PSD_Train,PSD_Val,PSD_Test}/{specular,diffuse}/<name>.png
    $DATASET_DIR/SHIQ/{train,test}/{diffuse,highlight}/<id>.png
    $DATASET_DIR/SSHR/{train,test}_<scene>/{rgb,diffuse}/<idx>.jpg

No file content is copied: PSD is exposed via a directory junction (its raw
layout already matches, it just needs "specular"/"diffuse" aliases), and
SHIQ/SSHR are relinked file-by-file with os.link (hardlinks, same volume,
zero extra disk usage).

SHIQ mapping note (not published by the upstream repo -- Test/Evaluate wiki
pages are marked WIP as of writing): the raw SHIQ quadruple is
_A=original/highlighted, _D=diffuse ground truth, _S=specular residue,
_T=binary highlight mask. wrappers.py's SHIQ entry sets RGB_DIR_NAME to
"diffuse" (the folder the loader treats as its "rgb" input) and pairs it
with a "highlight" mask folder -- which only makes sense for validating a
highlight *detector* if that folder actually holds the highlighted images.
So here "diffuse/" is populated from _A (not _D) and "highlight/" from _T.
Flag and revisit if the authors publish their real prep script.

SSHR has no wrapper in this repo yet; one is added to DATASET_DEFAULTS in
wrappers.py by this script's companion edit. GT diffuse uses _d_tc (tone
corrected to match input exposure) rather than _d.
"""
import argparse
import os
import subprocess
import sys


def _hardlink(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    os.link(src, dst)


def _junction(link: str, target: str) -> None:
    if os.path.exists(link):
        return
    os.makedirs(os.path.dirname(link), exist_ok=True)
    link = os.path.normpath(link)
    target = os.path.normpath(target)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", link, target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mklink failed for {link} -> {target}: {result.stdout} {result.stderr}")


def prepare_psd(raw_root: str, dataset_dir: str) -> None:
    """raw_root: .../extracted/PSD_raw (contains PSD_Dataset/...)"""
    psd_raw = os.path.join(raw_root, "PSD_Dataset")
    psd_out = os.path.join(dataset_dir, "PSD_Dataset")
    os.makedirs(psd_out, exist_ok=True)
    splits = {"PSD_Train": "PSD_Train", "PSD_val": "PSD_Val", "PSD_Test": "PSD_Test"}
    for raw_split, out_split in splits.items():
        raw_split_dir = os.path.join(psd_raw, raw_split)
        out_split_dir = os.path.join(psd_out, out_split)
        os.makedirs(out_split_dir, exist_ok=True)
        _junction(os.path.join(out_split_dir, "specular"), os.path.join(raw_split_dir, f"{raw_split}_specular"))
        _junction(os.path.join(out_split_dir, "diffuse"), os.path.join(raw_split_dir, f"{raw_split}_diffuse"))
    print(f"[PSD] ready at {psd_out}")


def prepare_shiq(raw_root: str, dataset_dir: str) -> None:
    """raw_root: .../extracted/SHIQ_raw (contains SHIQ_data_10825/{train,test})"""
    shiq_raw = None
    for name in os.listdir(raw_root):
        if name.startswith("SHIQ_data"):
            shiq_raw = os.path.join(raw_root, name)
            break
    if shiq_raw is None:
        raise FileNotFoundError(f"No SHIQ_data_* folder found under {raw_root}")

    shiq_out = os.path.join(dataset_dir, "SHIQ")
    for split in ("train", "test"):
        split_dir = os.path.join(shiq_raw, split)
        if not os.path.isdir(split_dir):
            continue
        count = 0
        for fname in os.listdir(split_dir):
            if fname.endswith("_A.png"):
                base = fname[: -len("_A.png")]
                _hardlink(
                    os.path.join(split_dir, fname),
                    os.path.join(shiq_out, split, "diffuse", f"{base}.png"),
                )
                t_name = f"{base}_T.png"
                if os.path.exists(os.path.join(split_dir, t_name)):
                    _hardlink(
                        os.path.join(split_dir, t_name),
                        os.path.join(shiq_out, split, "highlight", f"{base}.png"),
                    )
                # True diffuse ground truth (_D), for accuracy-metric evaluation only --
                # not used by the training loader, which reads "diffuse/" (see above) as
                # its rgb/input folder.
                d_name = f"{base}_D.png"
                if os.path.exists(os.path.join(split_dir, d_name)):
                    _hardlink(
                        os.path.join(split_dir, d_name),
                        os.path.join(shiq_out, split, "gt_clean", f"{base}.png"),
                    )
                count += 1
        print(f"[SHIQ] {split}: {count} images linked")
    print(f"[SHIQ] ready at {shiq_out}")


def prepare_sshr(raw_root: str, dataset_dir: str) -> None:
    """raw_root: .../extracted/SSHR_raw (contains SSHR/{train,test}/<scene>/<idx>_x.jpg)"""
    sshr_raw = os.path.join(raw_root, "SSHR")
    sshr_out = os.path.join(dataset_dir, "SSHR")
    for split in ("train", "test"):
        split_dir = os.path.join(sshr_raw, split)
        if not os.path.isdir(split_dir):
            continue
        for scene in os.listdir(split_dir):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            out_scene = f"{split}_{scene}"
            for fname in os.listdir(scene_dir):
                if fname.endswith("_i.jpg"):
                    base = fname[: -len("_i.jpg")]
                    _hardlink(
                        os.path.join(scene_dir, fname),
                        os.path.join(sshr_out, out_scene, "rgb", f"{base}.jpg"),
                    )
                    tc_name = f"{base}_d_tc.jpg"
                    if os.path.exists(os.path.join(scene_dir, tc_name)):
                        _hardlink(
                            os.path.join(scene_dir, tc_name),
                            os.path.join(sshr_out, out_scene, "diffuse", f"{base}.jpg"),
                        )
        print(f"[SSHR] {split} scenes linked")
    print(f"[SSHR] ready at {sshr_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True, help="Root folder holding PSD_raw/ SHIQ_raw/ SSHR_raw/")
    ap.add_argument("--dataset-dir", required=True, help="Target $DATASET_DIR")
    ap.add_argument("--datasets", nargs="+", default=["psd", "shiq", "sshr"], choices=["psd", "shiq", "sshr"])
    args = ap.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)
    if "psd" in args.datasets:
        prepare_psd(os.path.join(args.raw_dir, "PSD_raw"), args.dataset_dir)
    if "shiq" in args.datasets:
        prepare_shiq(os.path.join(args.raw_dir, "SHIQ_raw"), args.dataset_dir)
    if "sshr" in args.datasets:
        prepare_sshr(os.path.join(args.raw_dir, "SSHR_raw"), args.dataset_dir)
