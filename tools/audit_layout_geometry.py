"""Describe and draw the collection layouts exported by Unity."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


def coordinates(field):
    if not field.strip():
        return np.empty((0, 0))
    return np.array([[float(value) for value in group.split(",")] for group in field.strip().split(";")])


def audit_layouts(source):
    layouts = []
    skipped = []
    for line in source.read_text().splitlines():
        identifier, family, *geometry = line.split(maxsplit=2)
        if family == "skipped":
            skipped.append(identifier)
            continue
        doors, walls, boxes, reference, detour = geometry[0].split("|")
        layouts.append((identifier, family, int(doors),
                        coordinates(walls), coordinates(boxes),
                        coordinates(reference), coordinates(detour)))
    return layouts, skipped


def summary(layouts, skipped):
    turns = []
    extra = []
    closest = []
    headings = Counter()
    for _, _, _, _, boxes, reference, detour in layouts:
        if not len(detour):
            continue
        legs = np.diff(detour, axis=0)
        bearing = np.arctan2(legs[:, 0], legs[:, 1])
        changes = (np.diff(bearing) + np.pi) % (2 * np.pi) - np.pi
        turns.append(int(np.sum(np.abs(np.degrees(changes)) > 35)))
        extra.append(float(sum(np.linalg.norm(legs, axis=1)) - sum(np.linalg.norm(np.diff(reference, axis=0), axis=1))))
        if len(boxes) > 1:
            delta = np.maximum(boxes[:, None, :2] - (boxes[None, :, :2] + boxes[None, :, 2:]),
                               boxes[None, :, :2] - (boxes[:, None, :2] + boxes[:, None, 2:]))
            distances = np.linalg.norm(np.maximum(delta, 0), axis=2)
            np.fill_diagonal(distances, np.inf)
            closest.extend(distances.min(axis=1).tolist())
        headings[("north", "east", "south", "west")[round(float(bearing[0]) / (np.pi / 2)) % 4]] += 1

    def distribution(values):
        if not values:
            return None
        return dict(zip(("min", "median", "max"), np.quantile(values, (0, 0.5, 1)).round(3).tolist()))

    halls = [layout for layout in layouts if layout[1].startswith("hall-")]
    return {
        "layouts": len(layouts),
        "distinct_wall_geometries": len({layout[3].tobytes() for layout in layouts}),
        "skipped_cases": skipped,
        "families": dict(Counter(layout[1] for layout in layouts)),
        "doors": dict(sorted(Counter(layout[2] for layout in halls).items())),
        "initial_detour_heading": dict(headings),
        "boxes_per_layout": distribution([len(layout[4]) for layout in layouts]),
        "nearest_box_gap_m": distribution(closest),
        "extra_detour_length_m": distribution(extra),
        "detour_turns_over_35_deg": dict(sorted(Counter(turns).items())),
    }


def graph_audit(path):
    if not path.is_file():
        return {}
    with path.open(newline="") as file:
        return {row["seed"]: row for row in csv.DictReader(file)}


def graph_wall_edges(walls):
    starts = walls[:, :2]
    ends = np.round(starts + walls[:, 2:], 3)
    x, z = (np.unique(np.concatenate((starts[:, axis], ends[:, axis]))) for axis in (0, 1))
    x_starts, z_starts = np.searchsorted(x, starts[:, 0]), np.searchsorted(z, starts[:, 1])
    x_ends, z_ends = np.searchsorted(x, ends[:, 0]), np.searchsorted(z, ends[:, 1])
    occupied = np.zeros((len(x) - 1, len(z) - 1), dtype=bool)
    for x0, x1, z0, z1 in zip(x_starts, x_ends, z_starts, z_ends):
        occupied[x0:x1, z0:z1] = True

    # Draw only interfaces inside the map. Its outer rectangle is the export extent.
    edges = []
    vertical = occupied[:-1, :] != occupied[1:, :]
    horizontal = occupied[:, :-1].T != occupied[:, 1:].T
    for fixed, intervals in zip(x[1:-1], vertical):
        endpoints = np.flatnonzero(np.diff(np.r_[False, intervals, False])).reshape(-1, 2)
        edges.extend(((fixed, z[start]), (fixed, z[end])) for start, end in endpoints)
    for fixed, intervals in zip(z[1:-1], horizontal):
        endpoints = np.flatnonzero(np.diff(np.r_[False, intervals, False])).reshape(-1, 2)
        edges.extend(((x[start], fixed), (x[end], fixed)) for start, end in endpoints)
    return edges


def draw(layouts, graph_rows, columns, tile_size):
    rows = math.ceil(len(layouts) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(tile_size * columns, max(4, tile_size * rows)), squeeze=False)
    labels = {
        "network": "Corridor network", "rooms": "Narrow rooms", "hall-chain": "Linked rooms",
        "hall-wings": "Hall with wings", "hall-partitions": "Partitioned hall", "hall-atrium": "Atrium",
    }
    for ax, (identifier, family, doors, walls, boxes, reference, detour) in zip(axes.flat, layouts):
        graph_layout = family in ("network", "rooms")
        if graph_layout:
            ax.add_collection(LineCollection(graph_wall_edges(walls), colors="0.20", linewidths=0.7))
        else:
            for x, z, width, height in walls:
                ax.add_patch(Rectangle((x, z), width, height, fc="0.20", ec="none"))
        for x, z, width, height in boxes:
            ax.add_patch(Rectangle((x, z), width, height, fc="0.56", ec="none"))
        if len(detour):
            ax.plot(*reference.T, color="0.45", ls="--", lw=0.7)
            ax.plot(*detour.T, color="#175ea8", lw=0.95)
            ax.plot(*detour[0], marker="o", color="#175ea8", ms=1.8)
        if graph_layout:
            seed = identifier.split("/")[0]
            graph = graph_rows[seed]
            feasible = graph["feasible_pairs"]
            attempted = graph["attempted_pairs"]
            title = f"{seed}  {labels[family]}\n{feasible}/{attempted} feasible detours"
            scale = 5 if family == "network" else 3
        else:
            title = f"{identifier}  {labels[family]}\n{doors} entries, {len(boxes)} obstacles"
            scale = 2
        center = (walls[:, :2].min(axis=0) + (walls[:, :2] + walls[:, 2:]).max(axis=0)) / 2
        size = {"network": 42, "rooms": 20}.get(family, 17)
        half = size / 2
        ax.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half),
               aspect="equal", title=title)
        ax.plot([center[0] - half + size * 0.08, center[0] - half + size * 0.08 + scale],
                [center[1] - half + size * 0.05] * 2, color="0.35", lw=0.85)
        ax.title.set_fontsize(7 if columns >= 8 else 9)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.legend(handles=[
        Line2D([], [], color="#175ea8", label="Planned detour"),
        Line2D([], [], color="0.45", ls="--", label="Reference path"),
        Rectangle((0, 0), 1, 1, fc="0.56", ec="none", label="Obstacle"),
    ], loc="lower center", ncols=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.08 if columns < 8 else 0.03,
                        top=0.82 if columns < 8 else 0.97, wspace=0.12, hspace=0.37)
    return fig


def draw_pdf(layouts, graph_rows, output):
    graphs = [layout for layout in layouts if layout[1] == "network"]
    rooms = [layout for layout in layouts if layout[1] == "rooms"]
    halls = [layout for layout in layouts if layout[1].startswith("hall-")]
    pages = [(graphs[i:i + 10], 5, "Corridor networks", 5) for i in range(0, len(graphs), 10)]
    pages += [(rooms[i:i + 10], 5, "Narrow room maps", 3) for i in range(0, len(rooms), 10)]
    pages += [(halls[i:i + 12], 4, "Halls and linked rooms", 2) for i in range(0, len(halls), 12)]
    with PdfPages(output, metadata={"Title": "Unity v12 collection layouts"}) as pdf:
        for page, (items, columns, title, scale) in enumerate(pages, 1):
            fig = draw(items, graph_rows, columns, 2.8)
            fig.text(0.03, 0.955, title, size=14, va="top")
            fig.text(0.03, 0.914, f"Unity v12  |  Scale bars: {scale} m  |  Blue: planned detour", size=8, va="top")
            fig.text(0.97, 0.955, f"{page} / {len(pages)}", ha="right", va="top", size=9)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Unity output directory containing layouts.txt")
    args = parser.parse_args()
    layouts, skipped = audit_layouts(args.run / "layouts.txt")
    if not layouts:
        parser.error("Unity exported no usable layouts")
    graph_rows = graph_audit(args.run / "graph-audit.csv")
    stats = summary(layouts, skipped)
    if graph_rows:
        stats["graph_detour_checks"] = {}
        for family in ("network", "rooms"):
            rows = [row for row in graph_rows.values() if row["family"] == family]
            stats["graph_detour_checks"][family] = {
                "maps": len(rows),
                "feasible_pairs": sum(int(row["feasible_pairs"]) for row in rows),
                "attempted_pairs": sum(int(row["attempted_pairs"]) for row in rows),
                "maps_without_a_feasible_pair": sum(int(row["feasible_pairs"]) == 0 for row in rows),
            }
    report = json.dumps(stats, indent=2) + "\n"
    (args.run / "coverage.json").write_text(report)
    layouts.sort(key=lambda layout: (layout[1] not in ("network", "rooms"),
                                     layout[1] == "rooms", layout[0]))
    fig = draw(layouts, graph_rows, columns=10, tile_size=1.85)
    fig.savefig(args.run / "layouts-overview.png", dpi=180)
    plt.close(fig)
    if graph_rows:
        draw_pdf(layouts, graph_rows, args.run / f"floorplans-{len(layouts)}.pdf")
    print(report, end="")


if __name__ == "__main__":
    main()
