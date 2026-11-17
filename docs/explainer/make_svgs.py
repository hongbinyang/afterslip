#!/usr/bin/env python3
"""Generate the self-animating SVG figures for docs/explainer.md.

Pure SMIL/CSS animation (no JavaScript), so they play inside Markdown on
GitHub, VS Code preview and browsers. Stdlib only. Run from any directory:

    python docs/explainer/make_svgs.py

All figures are illustrations of ideas and hypotheses, not results.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent

STYLE = """<style>
.bg{fill:#eef2f6}.ink{fill:#1c1c1a}.mut{fill:#6b6a64}.ln{stroke:#c9c4b8}
.robot{fill:#2f5d8a}.robotS{stroke:#2f5d8a}.good{fill:#3f8f5a}.goodS{stroke:#3f8f5a}
.warn{fill:#c0503d}.gold{fill:#d39a2c}.goldS{stroke:#d39a2c}.acc{fill:#6a4c93}.accS{stroke:#6a4c93}
.demo{fill:rgba(106,76,147,.22)}.demoS{stroke:rgba(106,76,147,.5)}.card{fill:#ffffff}.floor{fill:#d9d4c7}
text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
@media (prefers-color-scheme: dark){
.bg{fill:#1e262e}.ink{fill:#ecebe5}.mut{fill:#a3a199}.ln{stroke:#4a483f}
.robot{fill:#7fb0de}.robotS{stroke:#7fb0de}.good{fill:#6fc28b}.goodS{stroke:#6fc28b}
.warn{fill:#e47e6c}.gold{fill:#e8bd62}.goldS{stroke:#e8bd62}.acc{fill:#b79be0}.accS{stroke:#b79be0}
.demo{fill:rgba(183,155,224,.25)}.demoS{stroke:rgba(183,155,224,.55)}.card{fill:#2a2a27}.floor{fill:#3a3730}}
</style>"""


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}"><title>{title}</title>{STYLE}'
            f'<rect class="bg" width="{w}" height="{h}" rx="10"/>{body}</svg>')


def anim(attr, values, kt, dur, extra=""):
    v = ";".join(str(x) for x in values)
    k = ";".join(f"{x:g}" for x in kt)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{dur}s" repeatCount="indefinite" {extra}/>'


def txt(x, y, s, cls="ink", size=13, weight="", anchor="start", extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}"{w} text-anchor="{anchor}">{s}{extra}</text>'


# ---------- 00: the whole story in one loop (before / after RL) ----------
def arm_panel(ox, recover):
    """One panel: a three-bin recycling sorter. The arm picks the metal item (gold),
    it slips; imitation-only freezes, the RL-practised arm re-grasps and sorts it."""
    D, kt = 9, [0, .12, .32, .42, .5, .62, .82, .92, 1]
    floor = 200
    if recover:
        tip = [(130, 70), (64, 168), (165, 88), (175, 88), (175, 88), (176, 168), (280, 104), (280, 104), (280, 104)]
        can = [(57, 178), (57, 178), (158, 98), (168, 98), (169, 178), (169, 178), (273, 114), (273, 172), (273, 172)]
    else:
        tip = [(130, 70), (64, 168), (165, 88), (175, 88), (175, 88), (182, 92), (168, 86), (178, 90), (175, 88)]
        can = [(57, 178), (57, 178), (158, 98), (168, 98), (169, 178), (169, 178), (169, 178), (169, 178), (169, 178)]
    tx = [ox + p[0] for p in tip]; ty = [p[1] for p in tip]
    cx = [ox + p[0] for p in can]; cy = [p[1] for p in can]
    b = []
    b.append(f'<rect class="floor" x="{ox+10}" y="{floor}" width="340" height="40"/>')
    # three bins: plastic (green), metal (gold), paper (purple)
    for bx, cls, name in ((228, "good", "plastic"), (280, "gold", "metal"), (332, "acc", "paper")):
        b.append(f'<rect class="{cls}" x="{ox+bx-22}" y="{floor-34}" width="44" height="34" opacity=".85"/>')
        b.append(txt(ox + bx, floor + 26, name, "mut", 11, anchor="middle"))
    # the other two items waiting on the table
    b.append(f'<rect class="good" width="14" height="22" x="{ox+22}" y="{floor-22}"/>')
    b.append(f'<rect class="acc" width="14" height="22" x="{ox+80}" y="{floor-22}"/>')
    b.append(f'<rect class="robot" x="{ox+115}" y="{floor-18}" width="30" height="18"/>')
    b.append(f'<line x1="{ox+130}" y1="{floor-18}" x2="{tx[0]}" y2="{ty[0]}" class="robotS" stroke-width="7" stroke-linecap="round">'
             + anim("x2", tx, kt, D) + anim("y2", ty, kt, D) + '</line>')
    b.append(f'<circle r="8" class="robot" cx="{tx[0]}" cy="{ty[0]}">' + anim("cx", tx, kt, D) + anim("cy", ty, kt, D) + '</circle>')
    b.append(f'<rect class="gold" width="14" height="22" x="{cx[0]}" y="{cy[0]}">' + anim("x", cx, kt, D) + anim("y", cy, kt, D) + '</rect>')
    # slip label
    b.append(f'<g opacity="0">{txt(ox+190, 150, "slip!", "warn", 14, "600")}'
             + anim("opacity", [0, 0, 1, 1, 0], [0, .42, .45, .58, .62], D) + '</g>')
    if recover:
        b.append(f'<g opacity="0">{txt(ox+180, 40, "re-grasps, sorts into metal ✓", "good", 14, "600", "middle")}'
                 + anim("opacity", [0, 0, 1, 1], [0, .82, .86, 1], D) + '</g>')
    else:
        b.append(f'<g opacity="0">{txt(ox+180, 40, "stuck: no demo ever showed this", "warn", 14, "600", "middle")}'
                 + anim("opacity", [0, 0, 1, 1], [0, .6, .65, 1], D) + '</g>')
    head = "Imitation only" if not recover else "After RL practice"
    b.append(txt(ox + 180, 20, head, "ink", 15, "600", "middle"))
    return "".join(b)


def fig00():
    body = arm_panel(0, False) + arm_panel(370, True)
    body += '<line x1="365" y1="10" x2="365" y2="250" class="ln" stroke-width="1"/>'
    # meters (hypothesis)
    y = 268
    def meter(ox, label, frac, cls):
        return (txt(ox, y, label, "mut", 12) + f'<rect x="{ox+110}" y="{y-10}" width="200" height="10" class="floor"/>'
                + f'<rect x="{ox+110}" y="{y-10}" width="{200*frac:.0f}" height="10" class="{cls}"/>')
    body += meter(20, "recovery", .12, "warn") + meter(390, "recovery", .8, "good")
    body += meter(20, "original skill", .9, "robot").replace(f'y="{y-10}"', f'y="{y+8}"').replace(f'y="{y}"', f'y="{y+18}"')
    body += meter(390, "original skill", .85, "robot").replace(f'y="{y-10}"', f'y="{y+8}"').replace(f'y="{y}"', f'y="{y+18}"')
    body += txt(730, 300, "hypothesis, not a result", "mut", 11, anchor="end")
    return svg(740, 310, body, "Before and after RL practice in a three-bin sorter: the arm drops a metal item; imitation-only freezes, the RL-practised arm re-grasps and sorts it")


# ---------- 01: learning by watching ----------
def fig01():
    D = 4.5
    b = '<rect class="floor" x="0" y="200" width="720" height="60"/>'
    b += '<rect class="gold" x="540" y="166" width="64" height="34"/>' + txt(572, 226, "bin", "mut", 12, anchor="middle")
    for j, hgt in enumerate([90, 105, 120, 135, 150]):
        b += f'<path d="M150 190 Q 360 {190-2*hgt} 572 160" fill="none" class="demoS" stroke-width="2" stroke-dasharray="5 5"/>'
    path = "M150 190 Q 360 -50 572 160"
    b += f'<path id="p1" d="{path}" fill="none"/>'
    b += ('<g><rect class="good" x="-8" y="-12" width="16" height="22"/><circle r="8" cy="-18" class="robot"/>'
          f'<animateMotion dur="{D}s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;.15;.85;1" calcMode="linear" path="{path}"/></g>')
    b += txt(20, 28, "dashed: human demos   ·   moving: the robot's learned motion", "mut", 12)
    return svg(720, 260, b, "Imitation learning: the robot copies demonstration paths into the bin")


# ---------- 02: the slip leaves the demo corridor ----------
def fig02():
    D = 6
    up = " ".join(f"{x},{130-28+30*__import__('math').sin(x/90):.1f}" for x in range(40, 690, 10))
    dn = " ".join(f"{x},{130+28+30*__import__('math').sin(x/90):.1f}" for x in range(680, 30, -10))
    b = f'<polygon points="{up} {dn}" class="demo"/>' + txt(48, 82, "demo corridor: where the robot has seen examples", "acc", 12)
    import math
    pts = []
    for x in range(40, 690, 10):
        y = 130 + 30 * math.sin(x / 90)
        if x > 360:
            d = (x - 360) / 320
            y += d * d * 120 + math.sin(x / 17) * 10 * d
        pts.append(f"{x},{y:.1f}")
    path = "M" + " L".join(pts)
    b += f'<path d="{path}" fill="none" class="robotS" stroke-width="3" stroke-dasharray="1200" stroke-dashoffset="1200">' \
         + anim("stroke-dashoffset", [1200, 0, 0], [0, .85, 1], D) + '</path>'
    b += f'<g opacity="0">{txt(345, 78, "slip!", "warn", 15, "600")}' + anim("opacity", [0, 0, 1, 1], [0, .42, .45, 1], D) + '</g>'
    b += f'<g opacity="0">{txt(520, 245, "?? no demo here", "warn", 14, "600")}' + anim("opacity", [0, 0, 1, 1], [0, .7, .75, 1], D) + '</g>'
    b += txt(690, 252, "state of the world →", "mut", 11, anchor="end")
    return svg(720, 260, b, "After a slip the robot's state leaves the region covered by demonstrations")


# ---------- 03: noise -> motion (flow matching) ----------
def fig03():
    import math, random
    random.seed(3)
    D = 5
    xs = [60 + i * 26 for i in range(24)]
    smooth = [150 - math.sin(i / 23 * math.pi) * 80 for i in range(24)]
    noisy = [120 + (random.random() * 2 - 1) * 70 for _ in range(24)]
    frames = []
    for s in range(6):
        a = s / 5
        a = a * a * (3 - 2 * a)
        frames.append(" ".join(f"{x},{(1-a)*n + a*m:.1f}" for x, n, m in zip(xs, noisy, smooth)))
    vals = frames + [frames[-1], frames[0]]
    kt = [0, .14, .28, .42, .56, .7, .95, 1]
    b = f'<polyline points="{frames[0]}" fill="none" class="robotS" stroke-width="3" stroke-linejoin="round">' \
        + f'<animate attributeName="points" values="{";".join(vals)}" keyTimes="{";".join(map(str, kt))}" dur="{D}s" repeatCount="indefinite"/></polyline>'
    for s in range(6):
        b += f'<circle cx="{250+s*44}" cy="28" r="6" class="floor"/>'
        b += f'<circle cx="{250+s*44}" cy="28" r="6" class="acc" opacity="0">' + anim("opacity", [0, 0, 1, 1, 0], [0, max(kt[s] - .001, 0), kt[s] if s else .001, .95, 1], D) + '</circle>'
    b += txt(40, 33, "refinement steps", "mut", 12)
    b += f'<g>{txt(60, 208, "random wiggles…", "mut", 13, "600")}' + anim("opacity", [1, 1, 0, 0, 1], [0, .6, .7, .95, 1], D) + '</g>'
    b += f'<g opacity="0">{txt(60, 208, "→ a smooth reach-and-place motion", "good", 13, "600")}' + anim("opacity", [0, 0, 1, 1, 0], [0, .65, .72, .95, 1], D) + '</g>'
    return svg(720, 220, b, "A flow-matching policy refines random noise into a smooth motion in a few steps")


# ---------- 04: trial and error ----------
def fig04():
    D = 8
    b = ""
    import random
    random.seed(4)
    for r in range(3):
        for c in range(8):
            x, y = 40 + c * 60, 36 + r * 62
            b += f'<rect x="{x}" y="{y}" width="50" height="50" class="card"/><rect x="{x}" y="{y}" width="50" height="50" fill="none" class="ln"/>'
            # each tile flashes red early more often, green later more often
            vals, kts = [], []
            for i in range(9):
                t = i / 8
                p_ok = .15 + .7 * t
                vals.append("good" if random.random() < p_ok else "warn")
            for cls in ("good", "warn"):
                ops = ["1" if v == cls else "0" for v in vals]
                b += f'<rect x="{x+15}" y="{y+15}" width="20" height="20" class="{cls}" opacity="0">' \
                     + f'<animate attributeName="opacity" values="{";".join(ops)}" dur="{D}s" repeatCount="indefinite" calcMode="discrete" begin="{(r*8+c)*0.07:.2f}s"/></rect>'
    bx, by, bh = 560, 210, 170
    b += txt(bx - 20, 24, "success on fumbles", "mut", 12)
    b += f'<rect x="{bx}" y="{by-bh}" width="60" height="{bh}" class="floor"/>'
    hs = [bh * v for v in (.15, .45, .65, .78, .85, .85)]
    b += f'<rect x="{bx}" width="60" class="good" y="{by-hs[0]}" height="{hs[0]}">' \
         + anim("height", [f"{h:.0f}" for h in hs], [0, .2, .45, .7, .9, 1], D) + anim("y", [f"{by-h:.0f}" for h in hs], [0, .2, .45, .7, .9, 1], D) + '</rect>'
    b += txt(40, 236, "many simulated attempts in parallel · green = recovered · red = failed · illustrative", "mut", 11)
    return svg(720, 245, b, "Reinforcement learning: many parallel attempts, success rate rising with practice")


# ---------- 05: the forgetting see-saw ----------
def fig05():
    D = 6
    b = '<polygon points="360,140 338,192 382,192" class="mut"/>'
    b += ('<g><rect x="100" y="134" width="520" height="8" class="ink"/>'
          '<rect x="115" y="84" width="110" height="50" class="good"/>' + txt(170, 115, "recovery", "card", 13, "600", "middle")
          + '<rect x="495" y="84" width="110" height="50" class="robot"/>' + txt(550, 115, "original skill", "card", 13, "600", "middle")
          + '<animateTransform attributeName="transform" type="rotate" values="12 360 138;-12 360 138;12 360 138" keyTimes="0;.5;1" dur="6s" repeatCount="indefinite"/></g>')
    b += txt(360, 214, "push one up, the other may go down", "mut", 13, anchor="middle")
    return svg(720, 225, b, "A see-saw between the new recovery skill and the original skill")


# ---------- 06: leash / freeze / rehearse ----------
def fig06():
    D = 6
    b = '<line x1="240" y1="16" x2="240" y2="214" class="ln"/><line x1="480" y1="16" x2="480" y2="214" class="ln"/>'
    ax, ay = 120, 110
    rx = [ax + 60, ax + 20, ax - 55, ax - 30, ax + 45, ax + 60]
    ry = [ay - 10, ay - 48, ay - 20, ay + 40, ay + 35, ay - 10]
    kt = [0, .2, .4, .6, .8, 1]
    b += f'<circle cx="{ax}" cy="{ay}" r="6" class="acc"/>' + txt(ax, ay + 24, "original", "acc", 11, anchor="middle")
    b += f'<line x1="{ax}" y1="{ay}" x2="{rx[0]}" y2="{ry[0]}" class="goldS" stroke-width="2">' + anim("x2", rx, kt, D) + anim("y2", ry, kt, D) + '</line>'
    b += f'<circle r="9" class="robot" cx="{rx[0]}" cy="{ry[0]}">' + anim("cx", rx, kt, D) + anim("cy", ry, kt, D) + '</circle>'
    b += txt(120, 205, "Leash", "ink", 15, "600", "middle")
    fx = 360
    for l in range(5):
        y = 34 + l * 30
        if l >= 3:
            b += f'<rect x="{fx-60}" y="{y}" width="120" height="20" class="robot" opacity=".5"/>' + txt(fx + 72, y + 15, "🔒", "ink", 13)
        else:
            b += f'<rect x="{fx-60}" y="{y}" width="120" height="20" class="gold"/>'
            b += f'<rect x="{fx-60}" y="{y}" width="120" height="20" class="good">' + anim("opacity", [0, 1, 0], [0, .5, 1], 1.6 + l * .4) + '</rect>'
        b += ""
    b += txt(fx - 66, 64, "learns", "mut", 11, anchor="end") + txt(fx - 66, 140, "locked", "mut", 11, anchor="end")
    b += txt(fx, 205, "Freeze", "ink", 15, "600", "middle")
    hx = 600
    b += f'<rect x="{hx-100}" y="70" width="92" height="60" class="acc">' + anim("opacity", [1, 1, .3, .3, 1], [0, .45, .5, .95, 1], 2.4) + '</rect>'
    b += txt(hx - 54, 105, "old demo", "ink", 12, "600", "middle")
    b += f'<rect x="{hx+8}" y="70" width="92" height="60" class="good">' + anim("opacity", [.3, .3, 1, 1, .3], [0, .45, .5, .95, 1], 2.4) + '</rect>'
    b += txt(hx + 54, 105, "new fumble", "ink", 12, "600", "middle")
    b += txt(hx, 156, "alternating while practising", "mut", 11, anchor="middle")
    b += txt(hx, 205, "Rehearse", "ink", 15, "600", "middle")
    return svg(720, 225, b, "Three mitigations: a leash to the original policy, frozen lower layers, rehearsing old demos")


# ---------- 07: fair test and trade-off plot ----------
def fig07():
    D = 9
    b = txt(24, 30, "the same saved fumbles for every robot:", "mut", 12)
    for i in range(8):
        x = 270 + i * 34
        b += f'<rect x="{x}" y="15" width="28" height="22" class="card"/><rect x="{x}" y="15" width="28" height="22" fill="none" class="ln"/>'
        b += f'<rect x="{x}" y="15" width="28" height="22" class="acc" opacity="0">' \
             + f'<animate attributeName="opacity" values="0;1;0;0" keyTimes="0;.04;.1;1" dur="2.4s" begin="{i*0.3:.1f}s" repeatCount="indefinite"/></rect>'
    ox, oy, W, H = 90, 262, 560, 190
    b += f'<line x1="{ox}" y1="{oy}" x2="{ox+W}" y2="{oy}" class="ln" stroke-width="1.5"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-H}" class="ln" stroke-width="1.5"/>'
    b += txt(ox + W, oy + 24, "original skill kept →", "mut", 12, anchor="end")
    b += f'<text x="{ox-16}" y="{oy}" class="mut" font-size="12" transform="rotate(-90 {ox-16} {oy})">recovery →</text>'
    b += f'<rect x="{ox+W*.75}" y="{oy-H}" width="{W*.25}" height="{H*.3}" class="demo"/>' + txt(ox + W * .875, oy - H + 18, "goal", "acc", 12, "600", "middle")
    pts = [("base robot", .86, .12, "mut"), ("RL, no help", .62, .74, "warn"), ("leash", .80, .66, "gold"),
           ("freeze", .76, .56, "robot"), ("rehearse", .83, .70, "good")]
    for i, (n, x, y, cls) in enumerate(pts):
        px, py = ox + x * W, oy - y * H
        t0 = .1 + i * .14
        b += f'<g opacity="0"><circle cx="{px}" cy="{py}" r="8" class="{cls}"/>{txt(px+12, py+4, n, "ink", 12)}' \
             + anim("opacity", [0, 0, 1, 1, 0], [0, t0, t0 + .03, .95, 1], D) + '</g>'
    b += txt(ox + 8, oy - H + 14, "hypothetical positions", "mut", 11)
    return svg(720, 300, b, "Hypothetical trade-off plot: recovery versus original skill kept, for each robot variant")


FIGS = {
    "00-whole-story.svg": fig00, "01-learning-by-watching.svg": fig01, "02-the-slip.svg": fig02,
    "03-noise-to-motion.svg": fig03, "04-trial-and-error.svg": fig04, "05-forgetting-seesaw.svg": fig05,
    "06-three-ways-to-remember.svg": fig06, "07-fair-test.svg": fig07,
}

if __name__ == "__main__":
    for name, fn in FIGS.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("wrote", name)
