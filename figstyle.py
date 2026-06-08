"""
공통 그림 스타일 — 모든 figure가 동일 팔레트/폰트를 쓰도록 중앙화.
analyze.py / analyze_nodes.py / analyze_gdpr.py 가 import.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 일관 팔레트
KO   = "#1f77b4"   # Korean (blue)
EN   = "#e8820c"   # English (orange)
NEG  = "#c0392b"   # negative gap / PIPA-specific / emphasis (red)
POS  = "#27ae60"   # positive gap (green)
BAR  = "#3b6fa0"   # neutral single-series bars (muted blue)
CMAP = "Blues"     # heatmaps

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11.5,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})
