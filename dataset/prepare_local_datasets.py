"""
Reorganizes freshly-downloaded PSD / SHIQ / SSHR archives into the folder
layout that UnReflectAnything_Dataset (see unreflectdataset.py + wrappers.py)
expects under a single $DATASET_DIR root:

    $DATASET_DIR/PSD_Dataset/{PSD_Train,PSD_Val,PSD_Test}/{specular,diffuse}/<name>.png
    $DATASET_DIR/SHIQ/{train,test}/{diffuse,highlight}/<id>.png
    $DATASET_DIR/SSHR/{train,test}_<scene>/{rgb,diffuse}/<idx>.jpg

No file content is copied: every file is relinked with os.link (hardlinks,
same volume, zero extra disk usage) rather than copied.

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


def _hardlink(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    os.link(src, dst)


def _normalize_stem(stem: str) -> str:
    """Strip punctuation/spaces and lowercase, so e.g. "Test (1)" and "Test_ (1)"
    (an inconsistency in the raw PSD_Test release -- specular/diffuse filenames
    there don't match verbatim, unlike PSD_Train/PSD_Val) compare equal. Digits
    are left untouched so "Val_0001" and "Val_01" still stay distinct (both
    genuinely occur in PSD_Val -- a naive int(digits) key would collide them)."""
    import re

    return re.sub(r"[^0-9a-zA-Z]", "", stem).lower()


def prepare_psd(raw_root: str, dataset_dir: str) -> None:
    """raw_root: .../extracted/PSD_raw (contains PSD_Dataset/...)

    Reads and re-links per file rather than aliasing whole folders: the raw
    release mixes .png/.JPG/.jpg extensions (wrappers.py's PSD entry hardcodes
    RGB_EXT=".png", so JPG-only files -- including the *entire* PSD_Test split
    -- would otherwise be silently invisible to the loader), and PSD_Test's
    specular/diffuse filenames don't match verbatim (see _normalize_stem).
    Output filenames are forced to a .png suffix (content is untouched, PIL
    sniffs the real codec) purely so RGB_EXT=".png" sees every file.
    """
    psd_raw = os.path.join(raw_root, "PSD_Dataset")
    psd_out = os.path.join(dataset_dir, "PSD_Dataset")
    splits = {"PSD_Train": "PSD_Train", "PSD_val": "PSD_Val", "PSD_Test": "PSD_Test"}
    for raw_split, out_split in splits.items():
        raw_split_dir = os.path.join(psd_raw, raw_split)
        specular_dir = os.path.join(raw_split_dir, f"{raw_split}_specular")
        diffuse_dir = os.path.join(raw_split_dir, f"{raw_split}_diffuse")
        if not os.path.isdir(specular_dir):
            continue

        diffuse_by_key = {}
        for fname in os.listdir(diffuse_dir):
            stem, _ = os.path.splitext(fname)
            diffuse_by_key[_normalize_stem(stem)] = fname

        linked = 0
        for fname in os.listdir(specular_dir):
            stem, _ = os.path.splitext(fname)
            key = _normalize_stem(stem)
            diffuse_fname = diffuse_by_key.get(key)
            if diffuse_fname is None:
                continue
            out_name = f"{stem}.png"
            _hardlink(os.path.join(specular_dir, fname), os.path.join(psd_out, out_split, "specular", out_name))
            _hardlink(os.path.join(diffuse_dir, diffuse_fname), os.path.join(psd_out, out_split, "diffuse", out_name))
            linked += 1
        print(f"[PSD] {out_split}: {linked} pairs linked")
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
