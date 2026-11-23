"""Planar LiDAR preview. Run with python3 lidar_sim.py."""

from math import cos, pi, sin

ROOM_LENGTH = 6.0
ROOM_WIDTH = 2.4
ROBOT = (1.1, ROOM_WIDTH / 2)
BOX_SIZE = (0.55, 0.65)
BOX_START = (3.2, ROOM_WIDTH / 2)


def rectangle_edges(left, bottom, right, top):
    return (
        ((left, bottom), (right, bottom)),
        ((right, bottom), (right, top)),
        ((right, top), (left, top)),
        ((left, top), (left, bottom)),
    )


def box_bounds(center):
    x, y = center
    w, h = BOX_SIZE
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def scan_ranges(box, ray_count=360, origin=ROBOT):
    """Return ranges in metres, starting forward and increasing toward the left."""
    edges = rectangle_edges(0, 0, ROOM_LENGTH, ROOM_WIDTH)
    edges += rectangle_edges(*box_bounds(box))
    ox, oy = origin
    ranges = []
    for index in range(ray_count):
        angle = 2 * pi * index / ray_count
        dx, dy = cos(angle), sin(angle)
        nearest = float("inf")
        for (ax, ay), (bx, by) in edges:
            sx, sy = bx - ax, by - ay
            determinant = dx * sy - dy * sx
            if abs(determinant) < 1e-10:
                continue
            rx, ry = ax - ox, ay - oy
            distance = (rx * sy - ry * sx) / determinant
            along = (rx * dy - ry * dx) / determinant
            if distance > 0 and 0 <= along <= 1:
                nearest = min(nearest, distance)
        ranges.append(nearest)
    return ranges


class Preview:
    def __init__(self, figure):
        from matplotlib.patches import Rectangle

        self.figure = figure
        self.box = BOX_START
        self.drag_offset = None
        self.axes = figure.subplots()
        self.axes.set(xlim=(-0.15, ROOM_LENGTH + 0.15),
                      ylim=(-0.15, ROOM_WIDTH + 0.15),
                      xlabel="x [m]", ylabel="y [m]", aspect="equal")
        self.axes.add_patch(Rectangle((0, 0), ROOM_LENGTH, ROOM_WIDTH, fill=False))
        self.obstacle = Rectangle((0, 0), *BOX_SIZE, fill=False)
        self.axes.add_patch(self.obstacle)
        x, y = ROBOT
        self.axes.add_patch(Rectangle((x - 0.2, y - 0.15), 0.4, 0.3, fill=False))
        self.axes.annotate("", xy=(x + 0.16, y), xytext=(x, y),
                           arrowprops={"arrowstyle": "->", "linewidth": 0.8})
        self.hits, = self.axes.plot([], [], ".", color="tab:blue", markersize=2,
                                    linestyle="none")
        self.range_text = self.axes.set_title("", loc="left", fontsize=10)

        canvas = figure.canvas
        canvas.mpl_connect("button_press_event", self.start_drag)
        canvas.mpl_connect("motion_notify_event", self.drag)
        canvas.mpl_connect("button_release_event", self.end_drag)
        canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    def start_drag(self, event):
        if event.button != 1 or event.inaxes is not self.axes:
            return
        x, y = event.xdata, event.ydata
        left, bottom, right, top = box_bounds(self.box)
        if left <= x <= right and bottom <= y <= top:
            self.drag_offset = (x - self.box[0], y - self.box[1])

    def drag(self, event):
        if self.drag_offset is None or event.inaxes is not self.axes:
            return
        offset_x, offset_y = self.drag_offset
        w, h = BOX_SIZE
        self.box = (
            min(ROOM_LENGTH - w / 2,
                max(ROBOT[0] + 0.25 + w / 2, event.xdata - offset_x)),
            min(ROOM_WIDTH - h / 2, max(h / 2, event.ydata - offset_y)),
        )
        self.redraw()

    def end_drag(self, _event):
        self.drag_offset = None

    def on_key(self, event):
        if event.key == "r":
            self.drag_offset = None
            self.box = BOX_START
            self.redraw()

    def redraw(self):
        left, bottom, _, _ = box_bounds(self.box)
        self.obstacle.set_xy((left, bottom))
        ranges = scan_ranges(self.box)
        angles = [2 * pi * i / len(ranges) for i in range(len(ranges))]
        ox, oy = ROBOT
        self.hits.set_data(
            [ox + distance * cos(angle) for distance, angle in zip(ranges, angles)],
            [oy + distance * sin(angle) for distance, angle in zip(ranges, angles)],
        )
        self.range_text.set_text(f"Forward range {ranges[0]:.2f} m")
        self.figure.canvas.draw_idle()


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", help="save the initial plot without opening a window")
    args = parser.parse_args()

    if args.save:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 3.8), dpi=140, layout="constrained")
        FigureCanvasAgg(figure)
        Preview(figure)
        figure.savefig(args.save, dpi=200)
        return

    from matplotlib import pyplot as plt

    with plt.rc_context({"toolbar": "None"}):
        figure = plt.figure(figsize=(8, 3.8), dpi=120, layout="constrained")
    figure.canvas.manager.set_window_title("LiDAR scan")
    preview = Preview(figure)
    plt.show()
    # Keep the preview alive while Matplotlib dispatches its bound callbacks.
    del preview


if __name__ == "__main__":
    main()
