"""Report generation: markdown + CSV tables and stdlib-only SVG curves."""
from __future__ import annotations

import csv
from pathlib import Path


def write_table(rows, columns, path_md, path_csv=None):
    if not rows:
        return
    def fmt(v):
        if isinstance(v, dict):
            return json_compact(v)
        return str(v)

    def json_compact(v):
        import json
        return json.dumps(v, ensure_ascii=False)

    lines = ['| ' + ' | '.join(columns) + ' |',
             '|' + '|'.join(['---'] * len(columns)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(fmt(r.get(c, '')) for c in columns) + ' |')
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)
    Path(path_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if path_csv:
        with open(path_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for r in rows:
                w.writerow({c: json_compact(r.get(c, '')) if isinstance(r.get(c), dict)
                            else r.get(c, '') for c in columns})


def svg_line_chart(series, xlabel, ylabel, title, path, width=680, height=360):
    """series: list of (label, points) with points = list of (x, y)."""
    all_pts = [p for _, pts in series for p in pts]
    if not all_pts:
        return
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 0.1
    pad, left, top = 12, 56, 34
    plot_w, plot_h = width - left - pad, height - top - 40
    colors = ['#c0392b', '#2471a3', '#1e8449', '#8e44ad', '#d68910', '#16a085']

    def px(x):
        return left + (x - x0) / (x1 - x0) * plot_w

    def py(y):
        return top + (1 - (y - y0) / (y1 - y0)) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             f'font-family="Segoe UI, sans-serif">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" fill="#333">{title}</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#999"/>',
             f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#999"/>',
             f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12" fill="#555">{xlabel}</text>',
             f'<text x="14" y="{top+plot_h/2}" font-size="12" fill="#555" transform="rotate(-90 14 {top+plot_h/2})">{ylabel}</text>']
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        parts.append(f'<text x="{left-6}" y="{py(y)+4}" text-anchor="end" font-size="10" fill="#777">{y:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{py(y)}" x2="{left+plot_w}" y2="{py(y)}" stroke="#eee"/>')
    for i in range(5):
        x = x0 + (x1 - x0) * i / 4
        parts.append(f'<text x="{px(x)}" y="{top+plot_h+16}" text-anchor="middle" font-size="10" fill="#777">{x:.0f}</text>')
    for idx, (label, pts) in enumerate(series):
        color = colors[idx % len(colors)]
        d = ' '.join(f'L{px(x):.1f},{py(y):.1f}' for x, y in pts)
        d = 'M' + d[1:]
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>')
        lx = left + plot_w * (0.15 + 0.7 * (idx % 3) / 2)
        ly = top + 14 + 16 * (idx // 3)
        parts.append(f'<circle cx="{lx-12}" cy="{ly-4}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{lx}" y="{ly}" font-size="11" fill="#333">{label}</text>')
    parts.append('</svg>')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(parts), encoding='utf-8')
