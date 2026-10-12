"""Render the manuscript's implementation overview using standard plotting tools."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root = Path(__file__).resolve().parents[1]
output = root / "paper/figures"
output.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")

def box(x, y, text, color):
    ax.add_patch(FancyBboxPatch((x, y), .27, .19, boxstyle="round,pad=0.01", facecolor=color, edgecolor="#334155"))
    ax.text(x + .135, y + .095, text, ha="center", va="center", fontsize=12)

def arrow(start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5))

box(.02, .75, "Natural-language\nmission", "#eff6ff")
box(.365, .75, "Offline translation\nand validation", "#eff6ff")
box(.71, .75, "LTLf specification\nand monitor", "#eff6ff")
box(.02, .28, "PPO training /\nlearned policy", "#ecfdf5")
box(.365, .28, "Optional bounds gate\nG(in_bounds)", "#fef3c7")
box(.71, .28, "PyQuaticus\nsimulator", "#ecfdf5")
arrow((.29, .845), (.355, .845)); arrow((.635, .845), (.7, .845))
arrow((.29, .375), (.355, .375)); arrow((.635, .375), (.7, .375))
arrow((.845, .48), (.845, .74))
ax.text(.825, .61, "Grounded\npropositions", fontsize=10, ha="right")
ax.annotate("", xy=(.155, .48), xytext=(.78, .74), arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="angle,angleA=180,angleB=90,rad=8", color="#334155"))
ax.text(.39, .64, "Optional temporal-state\nfeatures and reward shaping", ha="center", fontsize=11)
ax.plot([.845, .845, .155], [.27, .08, .08], color="#334155", lw=1.5)
arrow((.155, .08), (.155, .27))
ax.text(.50, .13, "Environment observations and rewards", ha="center", fontsize=11)
ax.text(.50, .025, "Evaluation: captures, pickups, tags, violations, duration, interventions", ha="center", fontsize=10)
fig.tight_layout(pad=.2)
fig.savefig(output / "figure1_system_architecture.png", dpi=300)
fig.savefig(output / "figure1_system_architecture.svg")
plt.close(fig)
