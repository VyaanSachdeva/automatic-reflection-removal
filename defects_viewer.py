"""
Builds raindrop_project/Defects(all)_viewer.html: a local page for browsing the
CCTV test set (eval_defects.py) with the rainy input, the released DeRaindrop,
our model and -- once eval_defects.py --segment has run -- the training-free
segment-and-repair pipeline with the instances it found, side by side:
synchronised zoom and pan, before/after sliders, and each image's measurements.
Images load from the folders next to the page, so nothing is copied or uploaded.

    python defects_viewer.py     # then open the HTML file in a browser
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import quote

from raindrop_paths import RAINDROP_ROOT

SRC = RAINDROP_ROOT / "Defects(all)"
FOLDERS = {"base": "Defects(all)", "released": "Defects(all)_deraindrop_released", "ours": "Defects(all)_derained"}
SEG_FOLDERS = {"seg": ("Defects(all)_segment_repair", ".png"), "inst": ("Defects(all)_segment_repair_instances", ".jpg")}
METRICS = RAINDROP_ROOT / "Defects(all)_comparisons" / "metrics.csv"
SEG_METRICS = RAINDROP_ROOT / "Defects(all)_comparisons_segment_repair" / "metrics.csv"  # all three models
OUT = RAINDROP_ROOT / "Defects(all)_viewer.html"


def _url(folder: str, rel: Path) -> str:
    return "/".join(quote(part) for part in (folder, *rel.parts))


def build() -> Path:
    metrics = {}
    has_seg = all((RAINDROP_ROOT / f).exists() for f, _ in SEG_FOLDERS.values())
    source = SEG_METRICS if has_seg and SEG_METRICS.exists() else METRICS
    if source.exists():
        for r in csv.DictReader(open(source, encoding="utf-8")):
            metrics.setdefault((r["scenario"], r["image"]), {})[r["model"]] = {
                k: float(r[k]) for k in ("change", "sharpness_ratio", "contrast_ratio")}
    scenarios = {}
    for p in sorted(SRC.rglob("*.png")):
        rel = p.relative_to(SRC)
        scenario = " - ".join(rel.parent.parts)
        scenarios.setdefault(scenario, []).append({
            "name": p.stem,
            **{k: _url(f, rel) for k, f in FOLDERS.items()},
            **({k: _url(f, rel.with_suffix(ext)) for k, (f, ext) in SEG_FOLDERS.items()} if has_seg else {}),
            "metrics": metrics.get((scenario, p.name), {}),
        })
    def category(s: str) -> str:
        if s.startswith("ghosting"):
            return "Ghosting (day)" if " - dge" in s else "Ghosting (night)"
        return s.split(" - ")[0].capitalize()

    data = [{"scenario": s, "category": category(s),
             "label": " - ".join(s.split(" - ")[2:] if s.startswith("ghosting") else s.split(" - ")[1:]),
             "images": imgs} for s, imgs in scenarios.items()]
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__SEG__", json.dumps(has_seg))
    OUT.write_text(html, encoding="utf-8")
    return OUT


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CCTV Rain Viewer</title>
<style>
  :root { --bg:#0e1622; --panel:#152131; --line:#24344a; --text:#e6edf5; --muted:#8fa1b6; --accent:#55a9d6; --ours:#f26a3d; --seg:#4cd18a; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 "Segoe UI", system-ui, sans-serif; display:flex; height:100vh; overflow:hidden; }
  aside { width:300px; flex:none; background:var(--panel); border-right:1px solid var(--line); overflow-y:auto; padding:14px 0; }
  aside h1 { font-size:16px; margin:0 16px 4px; }
  aside .sub { color:var(--muted); font-size:12px; margin:0 16px 12px; }
  aside h2 { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin:16px 16px 6px; }
  aside button.sc { display:flex; justify-content:space-between; gap:8px; width:100%; text-align:left; background:none; border:0; color:var(--text); padding:6px 16px; cursor:pointer; font:inherit; }
  aside button.sc:hover { background:#1c2b3e; }
  aside button.sc.on { background:#22364f; box-shadow:inset 3px 0 0 var(--accent); }
  aside button.sc span.n { color:var(--muted); }
  main { flex:1; display:flex; flex-direction:column; min-width:0; }
  header { display:flex; align-items:center; gap:12px; padding:10px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header .title { font-weight:600; font-size:15px; margin-right:auto; }
  header .title small { color:var(--muted); font-weight:400; margin-left:8px; }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
  .seg button { background:none; border:0; color:var(--text); padding:6px 12px; cursor:pointer; font:inherit; }
  .seg button.on { background:var(--accent); color:#06121d; font-weight:600; }
  .nav button, .reset { background:#1c2b3e; border:1px solid var(--line); color:var(--text); border-radius:6px; padding:6px 10px; cursor:pointer; font:inherit; }
  #stage { flex:1; display:grid; gap:8px; padding:8px; min-height:0; }
  #stage.three { grid-template-columns:repeat(3, 1fr); }
  #stage.all { grid-template-columns:repeat(3, 1fr); grid-template-rows:1fr 1fr; }
  .legend { border:1px solid var(--line); border-radius:6px; padding:12px 14px; color:var(--muted); font-size:13px; overflow:auto; }
  .legend b { color:var(--text); }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
  #stage.one { grid-template-columns:1fr; }
  .pane { position:relative; overflow:hidden; background:#070d15; border:1px solid var(--line); border-radius:6px; cursor:grab; min-height:0; }
  .pane.drag { cursor:grabbing; }
  .pane img { position:absolute; top:0; left:0; transform-origin:0 0; user-select:none; -webkit-user-drag:none; }
  .pane .tag { position:absolute; left:8px; top:8px; background:rgba(7,13,21,.8); padding:3px 8px; border-radius:4px; font-size:12px; font-weight:600; z-index:3; }
  .pane .tag.ours { color:var(--ours); }
  .pane .tag.rel { color:var(--accent); }
  .pane .tag.seg { color:var(--seg); }
  .pane .stats { position:absolute; left:8px; bottom:8px; background:rgba(7,13,21,.8); padding:3px 8px; border-radius:4px; font-size:12px; color:var(--muted); z-index:3; }
  .clip { position:absolute; inset:0; overflow:hidden; z-index:1; }
  .divider { position:absolute; top:0; bottom:0; width:2px; background:#fff; z-index:2; cursor:ew-resize; }
  .divider::after { content:"\2194"; position:absolute; top:50%; left:-14px; width:30px; height:30px; margin-top:-15px; border-radius:50%; background:#fff; color:#06121d; text-align:center; line-height:30px; font-weight:700; }
  footer { display:flex; gap:6px; padding:8px; border-top:1px solid var(--line); overflow-x:auto; }
  footer img { height:56px; border:2px solid transparent; border-radius:4px; cursor:pointer; opacity:.6; }
  footer img.on { border-color:var(--accent); opacity:1; }
  .help { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<aside>
  <h1>CCTV rain test set</h1>
  <div class="sub" id="sub">Rainy input · DeRaindrop (released) · Ours (fine-tuned on VRGS)</div>
  <div id="list"></div>
</aside>
<main>
  <header>
    <div class="title" id="title"></div>
    <div class="seg" id="mode"></div>
    <button class="reset" id="reset" title="Reset zoom (0)">Reset zoom</button>
    <span class="nav"><button id="prev" title="Previous image (Left arrow)">&larr;</button> <span id="pos"></span> <button id="next" title="Next image (Right arrow)">&rarr;</button></span>
  </header>
  <div id="stage" class="three"></div>
  <footer id="strip"></footer>
  <div class="help" style="padding:0 12px 8px">Scroll to zoom · drag to pan (all panels move together) · &larr;/&rarr; image · &uarr;/&darr; scenario · number keys: view mode · 0 reset zoom</div>
</main>
<script>
const DATA = __DATA__, SEG = __SEG__;
const MODES = SEG ? [["all", "All models"], ["slider-seg", "Slider: input vs new"], ["slider-ours-seg", "Slider: ours vs new"],
                     ["slider-ours", "Slider: input vs ours"], ["slider-rel", "Slider: input vs released"], ["three", "Old three"]]
                  : [["three", "Side by side"], ["slider-ours", "Slider: input vs ours"], ["slider-rel", "Slider: input vs released"],
                     ["slider-both", "Slider: released vs ours"]];
let si = 0, ii = 0, mode = MODES[0][0];
let view = { s: 1, x: 0, y: 0 }, split = 0.5;
const fmt = m => m ? `change ${m.change.toFixed(1)} · sharp ×${m.sharpness_ratio.toFixed(2)} · contrast ×${m.contrast_ratio.toFixed(2)}` : "";

function buildList() {
  const list = document.getElementById("list"); let cat = null;
  DATA.forEach((d, i) => {
    if (d.category !== cat) { cat = d.category; const h = document.createElement("h2"); h.textContent = cat; list.appendChild(h); }
    const b = document.createElement("button"); b.className = "sc"; b.dataset.i = i;
    b.innerHTML = `<span></span><span class="n">${d.images.length}</span>`; b.firstChild.textContent = d.label || d.scenario;
    b.onclick = () => { si = i; ii = 0; render(); }; list.appendChild(b);
  });
}

function pane(src, tag, cls, stats) {
  const p = document.createElement("div"); p.className = "pane";
  p.innerHTML = `<img alt=""><div class="tag ${cls}"></div>` + (stats ? `<div class="stats"></div>` : "");
  p.querySelector("img").src = src; p.querySelector(".tag").textContent = tag;
  if (stats) p.querySelector(".stats").textContent = stats;
  return p;
}

function sliderPane(left, right, lTag, rTag, lCls, rCls) {
  const p = document.createElement("div"); p.className = "pane";
  p.innerHTML = `<img alt=""><div class="clip"><img alt=""></div><div class="divider"></div>
    <div class="tag ${lCls}"></div><div class="tag ${rCls}" style="left:auto;right:8px"></div>`;
  const [a, b] = p.querySelectorAll("img"); a.src = right; b.src = left;
  const tags = p.querySelectorAll(".tag"); tags[0].textContent = lTag; tags[1].textContent = rTag;
  const div = p.querySelector(".divider");
  div.addEventListener("pointerdown", e => {
    e.stopPropagation(); div.setPointerCapture(e.pointerId);
    const move = ev => { const r = p.getBoundingClientRect(); split = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)); applyView(); };
    div.addEventListener("pointermove", move);
    div.addEventListener("pointerup", () => div.removeEventListener("pointermove", move), { once: true });
  });
  return p;
}

function fit(pane, img) {
  const w = img.naturalWidth || 1, h = img.naturalHeight || 1, r = pane.getBoundingClientRect();
  return Math.min(r.width / w, r.height / h);
}

function applyView() {
  document.querySelectorAll("#stage .pane").forEach(p => {
    const imgs = p.querySelectorAll("img"); const base = imgs[0];
    if (!base.naturalWidth) return;
    const r = p.getBoundingClientRect(), f = fit(p, base);
    // Every image in a pane is drawn at the base image's displayed size, so outputs at a
    // different resolution line up pixel for pixel.
    const W = base.naturalWidth * f * view.s, H = base.naturalHeight * f * view.s;
    const ox = (r.width - base.naturalWidth * f) / 2 + view.x, oy = (r.height - base.naturalHeight * f) / 2 + view.y;
    imgs.forEach(im => { im.style.width = W + "px"; im.style.height = H + "px"; im.style.transform = `translate(${ox}px, ${oy}px)`; });
    const clip = p.querySelector(".clip"), div = p.querySelector(".divider");
    if (clip) { clip.style.clipPath = `inset(0 ${(1 - split) * 100}% 0 0)`; div.style.left = (split * r.width - 1) + "px"; }
  });
}

function render() {
  const d = DATA[si], im = d.images[ii];
  document.querySelectorAll("aside .sc").forEach(b => b.classList.toggle("on", +b.dataset.i === si));
  document.getElementById("title").innerHTML = ""; const t = document.getElementById("title");
  t.textContent = d.scenario; const sm = document.createElement("small"); sm.textContent = im.name; t.appendChild(sm);
  document.getElementById("pos").textContent = `${ii + 1} / ${d.images.length}`;
  const stage = document.getElementById("stage"); stage.innerHTML = "";
  stage.className = mode === "three" ? "three" : mode === "all" ? "all" : "one";
  const m = im.metrics || {};
  if (mode === "all") {
    const legend = document.createElement("div"); legend.className = "legend";
    legend.innerHTML = `<b>Instances found</b> (new pipeline, no training):<br><i style="background:#ffaa1e"></i>glare<br>
      <i style="background:#c85aff"></i>streak: light smeared on the dome, lit webs<br><i style="background:#28aaff"></i>raindrop<br>
      <i style="background:#3cdc5a"></i>person or vehicle, protected<br><i style="background:#96a0af"></i>wet road, kept as real scene (SAM 3)<br><br>
      <b>New: segment &amp; repair</b> = our DeRaindrop on the whole frame, then Flare7K++ takes the glare off,
      streaks are dimmed and the wet road is kept from the original.<br><br>No clean version of this footage exists, so the
      numbers describe what each model did: change (0-255), sharpness and contrast relative to the input.`;
    stage.append(pane(im.base, "Rainy input", "", ""), pane(im.inst, "Instances found", "seg", ""),
                 pane(im.released, "DeRaindrop (released)", "rel", fmt(m.released)),
                 pane(im.ours, "Ours: fine-tuned on VRGS", "ours", fmt(m.ours)),
                 pane(im.seg, "New: segment & repair", "seg", fmt(m.segment)), legend);
  } else if (mode === "slider-seg") stage.append(sliderPane(im.base, im.seg, "Rainy input", "New: segment & repair", "", "seg"));
  else if (mode === "slider-ours-seg") stage.append(sliderPane(im.ours, im.seg, "Ours", "New: segment & repair", "ours", "seg"));
  else if (mode === "three") {
    stage.append(pane(im.base, "Rainy input", "", ""), pane(im.released, "DeRaindrop (released)", "rel", fmt(m.released)),
                 pane(im.ours, "Ours: fine-tuned on VRGS", "ours", fmt(m.ours)));
  } else if (mode === "slider-ours") stage.append(sliderPane(im.base, im.ours, "Rainy input", "Ours", "", "ours"));
  else if (mode === "slider-rel") stage.append(sliderPane(im.base, im.released, "Rainy input", "Released", "", "rel"));
  else stage.append(sliderPane(im.released, im.ours, "Released", "Ours", "rel", "ours"));
  stage.querySelectorAll("img").forEach(i => i.addEventListener("load", applyView));
  const strip = document.getElementById("strip"); strip.innerHTML = "";
  d.images.forEach((x, k) => { const th = document.createElement("img"); th.src = x.base; th.loading = "lazy"; th.alt = x.name;
    th.className = k === ii ? "on" : ""; th.onclick = () => { ii = k; render(); }; strip.appendChild(th); });
  bindPan();
}

function bindPan() {
  document.querySelectorAll("#stage .pane").forEach(p => {
    p.addEventListener("wheel", e => {
      e.preventDefault(); const r = p.getBoundingClientRect(), k = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const ns = Math.min(12, Math.max(1, view.s * k)), ratio = ns / view.s;
      const cx = e.clientX - r.left - r.width / 2, cy = e.clientY - r.top - r.height / 2;
      view.x = cx - (cx - view.x) * ratio; view.y = cy - (cy - view.y) * ratio; view.s = ns;
      if (ns === 1) { view.x = 0; view.y = 0; } applyView();
    }, { passive: false });
    p.addEventListener("pointerdown", e => {
      if (e.target.classList.contains("divider")) return;
      p.classList.add("drag"); p.setPointerCapture(e.pointerId); let lx = e.clientX, ly = e.clientY;
      const move = ev => { view.x += ev.clientX - lx; view.y += ev.clientY - ly; lx = ev.clientX; ly = ev.clientY; applyView(); };
      p.addEventListener("pointermove", move);
      p.addEventListener("pointerup", () => { p.classList.remove("drag"); p.removeEventListener("pointermove", move); }, { once: true });
    });
  });
}

function step(di) { const n = DATA[si].images.length; ii = (ii + di + n) % n; render(); }
function stepScenario(d) { si = (si + d + DATA.length) % DATA.length; ii = 0; render(); }
function setMode(m) { mode = m; document.querySelectorAll("#mode button").forEach(b => b.classList.toggle("on", b.dataset.m === m)); render(); }

MODES.forEach(([m, label], k) => { const b = document.createElement("button"); b.dataset.m = m; b.textContent = label;
  b.title = `Key ${k + 1}`; b.className = k === 0 ? "on" : ""; b.onclick = () => setMode(m); document.getElementById("mode").appendChild(b); });
if (SEG) document.getElementById("sub").textContent = "Rainy input · instances found · DeRaindrop (released) · ours · new segment & repair";
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
document.getElementById("reset").onclick = () => { view = { s: 1, x: 0, y: 0 }; applyView(); };
window.addEventListener("resize", applyView);
window.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") step(1); else if (e.key === "ArrowLeft") step(-1);
  else if (e.key === "ArrowDown") { e.preventDefault(); stepScenario(1); } else if (e.key === "ArrowUp") { e.preventDefault(); stepScenario(-1); }
  else if (e.key === "0") { view = { s: 1, x: 0, y: 0 }; applyView(); }
  else if (e.key >= "1" && e.key <= String(MODES.length)) setMode(MODES[+e.key - 1][0]);
});
buildList(); render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Wrote {build()}")
