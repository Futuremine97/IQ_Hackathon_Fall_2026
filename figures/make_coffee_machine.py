"""README illustration -> figures/coffee_machine.svg

A cup of coffee goes into a machine that runs this repository's pipeline (Prophet forecast,
4-qubit quantum kernel, 18-qubit QAOA) and prints an equal-weight 4-ETF basket. The basket is
the QAOA pick of results/LIVE_2026-10-23.md and the receipt footer is the pre-registered
backtest result H4 (results/RESULT_QAOA.json). Hand-drawn SVG, no data plotted.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
H4 = json.load(open(os.path.join(ROOT, "results", "RESULT_QAOA.json")))["H4"]

INK, INK2, MUTED, PAPER, CARD = "#26251f", "#6b6a63", "#b5b4ad", "#f6f3ec", "#ffffff"
QC, CC, AQ, NEU = "#2a78d6", "#eb6834", "#1baf7a", "#8a897f"
BODY, COPPER, COPPER2, BEAN, BEAN2 = "#3a3a42", "#c8814a", "#a8683a", "#6b4226", "#3e2716"
SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, DejaVu Sans Mono, monospace"
BASKET = [("XLU", QC), ("EEM", CC), ("TLT", AQ), ("IEF", NEU)]

out = []
w = out.append


def text(x, y, s, size=13, fill=INK, weight="normal", anchor="start", family=SANS, extra=""):
    w(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
      f'fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')


w('<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="620" viewBox="0 0 1280 620" '
  'role="img" aria-labelledby="t d">')
w('<title id="t">The coffee-to-portfolio machine</title>')
w('<desc id="d">A cup of coffee is poured into a machine with three windows - Prophet forecast, '
  '4-qubit quantum kernel, 18-qubit QAOA - which prints an equal-weight four-ETF basket. '
  f'The receipt notes the backtest Sharpe ratio: QAOA {H4["sharpe_qaoa"]:.2f} versus equal weight {H4["sharpe_ew"]:.2f}.</desc>')
w('<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{NEU}"/></marker></defs>')
w(f'<rect x="0" y="0" width="1280" height="620" rx="24" fill="{PAPER}"/>')

# title
text(48, 60, "Coffee in, optimized portfolio out", 30, INK, "700")
text(48, 90, "Prophet → quantum kernel → QAOA, in one machine", 16, INK2)

# floor and shadows
w(f'<line x1="40" y1="540" x2="1240" y2="540" stroke="#ddd8cc" stroke-width="3" stroke-linecap="round"/>')
w(f'<ellipse cx="630" cy="541" rx="290" ry="9" fill="{INK}" opacity="0.08"/>')

# ---------------------------------------------------------------- cup (input)
for x0 in (128, 150, 172):
    dy = -6 if x0 == 150 else 0
    w(f'<path d="M {x0} {404 + dy} c -10 -14 10 -22 0 -36 c -10 -14 10 -22 0 -36" fill="none" '
      f'stroke="{MUTED}" stroke-width="3.5" stroke-linecap="round"/>')
w(f'<ellipse cx="150" cy="526" rx="88" ry="12" fill="#efe9dd" stroke="{INK}" stroke-width="3"/>')
w(f'<path d="M 210 440 C 254 438 254 496 202 490" fill="none" stroke="{INK}" stroke-width="13" stroke-linecap="round"/>')
w(f'<path d="M 210 440 C 254 438 254 496 202 490" fill="none" stroke="{CARD}" stroke-width="7" stroke-linecap="round"/>')
w(f'<path d="M 88 420 L 212 420 L 200 502 Q 197 522 177 522 L 123 522 Q 103 522 100 502 Z" '
  f'fill="{CARD}" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>')
w(f'<ellipse cx="150" cy="420" rx="62" ry="9" fill="{BEAN}" stroke="{INK}" stroke-width="3"/>')
w(f'<ellipse cx="146" cy="419" rx="40" ry="4.5" fill="#b07a45" opacity="0.75"/>')
text(150, 476, "|coffee⟩", 15, QC, "700", "middle", MONO)

# pour arrow
w(f'<path d="M 228 398 C 262 240 400 100 538 112" fill="none" stroke="{NEU}" stroke-width="2.5" '
  'stroke-dasharray="7 7" marker-end="url(#ah)"/>')

# ---------------------------------------------------------------- machine
# hopper with beans
w(f'<polygon points="590,176 670,176 712,120 548,120" fill="{COPPER}" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>')
w(f'<rect x="542" y="112" width="176" height="11" rx="4" fill="{COPPER2}" stroke="{INK}" stroke-width="3"/>')
for (bx, by, rot) in [(574, 104, -20), (598, 98, 15), (622, 105, -35), (646, 97, 25), (670, 104, -10), (692, 100, 30)]:
    w(f'<g transform="rotate({rot} {bx} {by})"><ellipse cx="{bx}" cy="{by}" rx="9" ry="6.2" fill="{BEAN}" '
      f'stroke="{BEAN2}" stroke-width="1.5"/><path d="M {bx - 6} {by} q 6 -3 12 0" fill="none" stroke="{BEAN2}" stroke-width="1.5"/></g>')

# feet, body, cap
for fx in (410, 800):
    w(f'<rect x="{fx}" y="496" width="50" height="42" rx="7" fill="{INK}"/>')
w(f'<rect x="360" y="204" width="540" height="300" rx="28" fill="{BODY}" stroke="{INK}" stroke-width="3"/>')
w(f'<rect x="380" y="176" width="500" height="34" rx="12" fill="{INK}"/>')
text(630, 199, "Q-BREW 18", 14, PAPER, "700", "middle", extra='letter-spacing="4"')
for cx, col in ((842, AQ), (858, CC)):
    w(f'<circle cx="{cx}" cy="193" r="4" fill="{col}"/>')

# pipes between the three stage windows
for px0 in (540, 710):
    w(f'<rect x="{px0 - 2}" y="296" width="24" height="12" fill="{COPPER}" stroke="{INK}" stroke-width="2"/>')

# window 1: Prophet forecast
w(f'<rect x="390" y="226" width="150" height="150" rx="12" fill="{PAPER}"/>')
text(402, 248, "1 · PROPHET", 12, INK, "700", extra='letter-spacing="1"')
w(f'<line x1="402" y1="352" x2="528" y2="352" stroke="{MUTED}" stroke-width="1.5"/>')
w(f'<polygon points="470,298 528,262 528,314" fill="{CC}" opacity="0.22"/>')
w(f'<polyline points="402,336 412,328 420,332 430,316 440,322 450,308 460,312 470,298" fill="none" '
  f'stroke="{INK}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
w(f'<line x1="470" y1="298" x2="528" y2="288" stroke="{CC}" stroke-width="2.2" stroke-dasharray="5 4"/>')
text(402, 369, "30-day forecast", 11, INK2)

# window 2: ZZ feature map (H on each qubit, ZZ on every pair)
w(f'<rect x="560" y="226" width="150" height="150" rx="12" fill="{PAPER}"/>')
text(572, 248, "2 · QSVM", 12, INK, "700", extra='letter-spacing="1"')
wy = [272, 292, 312, 332]
for y in wy:
    w(f'<line x1="574" y1="{y}" x2="698" y2="{y}" stroke="{INK}" stroke-width="1.5"/>')
    w(f'<rect x="580" y="{y - 8}" width="16" height="16" rx="3" fill="{QC}"/>')
    text(588, y + 4, "H", 10, CARD, "700", "middle")
for k, (i, j) in enumerate([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]):
    x = 612 + 14 * k
    w(f'<line x1="{x}" y1="{wy[i]}" x2="{x}" y2="{wy[j]}" stroke="{QC}" stroke-width="2"/>')
    for y in (wy[i], wy[j]):
        w(f'<circle cx="{x}" cy="{y}" r="3.6" fill="{QC}"/>')
text(572, 369, "4-qubit kernel", 11, INK2)

# window 3: QAOA register, 18 qubits, 4 selected
w(f'<rect x="730" y="226" width="150" height="150" rx="12" fill="{PAPER}"/>')
text(742, 248, "3 · QAOA", 12, INK, "700", extra='letter-spacing="1"')
lit = {(1, 0), (4, 0), (2, 1), (5, 2)}
for r in range(3):
    for c in range(6):
        x, y = 753 + 21 * c, 276 + 24 * r
        if (c, r) in lit:
            w(f'<circle cx="{x}" cy="{y}" r="10" fill="{QC}" opacity="0.18"/>')
            w(f'<circle cx="{x}" cy="{y}" r="6.5" fill="{QC}"/>')
        else:
            w(f'<circle cx="{x}" cy="{y}" r="6" fill="none" stroke="{NEU}" stroke-width="1.8"/>')
text(742, 369, "18 qubits → pick 4", 11, INK2)

# control strip: coffee preset = risk aversion
text(392, 400, "coffee setting = risk aversion", 11, MUTED)
for k, name in enumerate(["ESPRESSO", "AMERICANO", "LATTE", "DECAF"]):
    cx = 432 + 96 * k
    on = name == "AMERICANO"
    if on:
        w(f'<circle cx="{cx}" cy="425" r="20" fill="{CC}" opacity="0.25"/>')
    w(f'<circle cx="{cx}" cy="425" r="13" fill="{CC if on else "#5a5a62"}" stroke="{INK}" stroke-width="2.5"/>')
    text(cx, 460, name, 10, PAPER if on else MUTED, "700", "middle", extra='letter-spacing="1"')

# gauge
w(f'<circle cx="842" cy="438" r="34" fill="{PAPER}" stroke="{COPPER}" stroke-width="4"/>')
for a in range(0, 181, 30):
    t = math.radians(180 + a)
    x1, y1 = 842 + 26 * math.cos(t), 444 + 26 * math.sin(t)
    x2, y2 = 842 + 31 * math.cos(t), 444 + 31 * math.sin(t)
    w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="2"/>')
t = math.radians(180 + 150)
w(f'<line x1="842" y1="444" x2="{842 + 24 * math.cos(t):.1f}" y2="{444 + 24 * math.sin(t):.1f}" '
  f'stroke="{CC}" stroke-width="3" stroke-linecap="round"/>')
w(f'<circle cx="842" cy="444" r="4" fill="{INK}"/>')
text(842, 463, "QUBITS", 8, INK2, "700", "middle", extra='letter-spacing="1"')

# output pipe
w(f'<rect x="896" y="300" width="58" height="24" fill="{COPPER}" stroke="{INK}" stroke-width="3"/>')
w(f'<rect x="946" y="292" width="10" height="40" rx="2" fill="{COPPER2}" stroke="{INK}" stroke-width="3"/>')

# ---------------------------------------------------------------- receipt (output)
x0, x1, ytop, ybot = 960, 1236, 232, 506
teeth = 20
zig = " ".join(f"L {x1 - (x1 - x0) * i / teeth:.1f} {ybot if i % 2 == 0 else ybot - 8}" for i in range(teeth + 1))
card = f"M {x0} {ytop} L {x1} {ytop} {zig} Z"
w(f'<path d="{card}" transform="translate(4 5)" fill="{INK}" opacity="0.12"/>')
w(f'<path d="{card}" fill="{CARD}" stroke="{INK}" stroke-width="2" stroke-linejoin="round"/>')
text(1098, 260, "OPTIMIZED PORTFOLIO", 15, INK, "700", "middle", extra='letter-spacing="1.5"')
text(1098, 280, "QAOA top-4 · 2026-10-23 close", 12, INK2, anchor="middle")
w(f'<line x1="976" y1="294" x2="1220" y2="294" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4"/>')
cx, cy, r = 1030, 356, 42
circ = 2 * math.pi * r
gap = 4.0
for k, (_, col) in enumerate(BASKET):
    rot = -90 + 90 * k + math.degrees(gap / 2 / r)
    w(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="22" '
      f'stroke-dasharray="{circ / 4 - gap:.2f} {circ:.2f}" transform="rotate({rot:.2f} {cx} {cy})"/>')
for k, (tick, col) in enumerate(BASKET):
    y = 326 + 22 * k
    w(f'<rect x="1104" y="{y - 10}" width="11" height="11" rx="2" fill="{col}"/>')
    text(1122, y, tick, 14, INK, "700", family=MONO)
    text(1218, y, "25 %", 13, INK2, anchor="end")
w(f'<line x1="976" y1="420" x2="1220" y2="420" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4"/>')
text(1098, 442, "backtest Sharpe, 2014–2026", 12, INK2, anchor="middle")
text(1098, 464, f"QAOA {H4['sharpe_qaoa']:.2f} · equal weight {H4['sharpe_ew']:.2f}", 13, INK, "700", "middle")
text(1098, 486, "not investment advice", 11, INK2, anchor="middle", extra='font-style="italic"')

# ---------------------------------------------------------------- captions
for x, a, b in [(150, "INPUT", "one coffee"),
                (630, "MACHINE", "Prophet → QSVM → QAOA (exact simulation)"),
                (1098, "OUTPUT", "an optimized 4-ETF basket")]:
    text(x, 576, a, 12, NEU, "700", "middle", extra='letter-spacing="2"')
    text(x, 598, b, 15, INK, anchor="middle")
w("</svg>")

open(os.path.join(HERE, "coffee_machine.svg"), "w").write("\n".join(out) + "\n")
print("written figures/coffee_machine.svg")
