import matplotlib.pyplot as plt
import numpy as np
import json

from Dataset.DataSetLoaders import ChessDataset


class Annotator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

        self.edited = {}
        self.faulty = set()
        self.active_point = None

        # For panning
        self.press_event = None

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.show_image()

    # -------------------------------
    # LOADING + DISPLAY
    # -------------------------------
    def load_coords(self):
        img, y = self.dataset[self.index]
        coords = y["corners"]
        if self.index in self.edited:
            coords = np.array(self.edited[self.index])
        else:
            coords = np.array(coords)
        return img, coords

    def show_image(self):
        self.ax.clear()

        img, coords = self.load_coords()
        self.img = img
        self.coords = coords

        if hasattr(img, "permute"):
            img = img.permute(1, 2, 0).cpu().numpy()

        self.ax.imshow(img)
        self.scatter = self.ax.scatter(self.coords[:, 0], self.coords[:, 1],
                                       c='red', s=40)

        faulty = " — FAULTY" if self.index in self.faulty else ""
        self.ax.set_title(
            f"Image {self.index+1}/{len(self.dataset)}{faulty} | "
            "1–4 select, left-click move, scroll=zoom, right-drag=pan, "
            "n=next, b=back, f=faulty, q=quit"
        )
        self.fig.canvas.draw()

    def save_current(self):
        self.edited[self.index] = self.coords.tolist()

    # -------------------------------
    # ZOOM
    # -------------------------------
    def on_scroll(self, event):
        """Mouse wheel zoom."""
        if event.xdata is None or event.ydata is None:
            return

        scale = 1.15 if event.button == 'up' else 1 / 1.15

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Zoom relative to mouse pointer
        self.ax.set_xlim([event.xdata - (event.xdata - xlim[0]) * scale,
                          event.xdata + (xlim[1] - event.xdata) * scale])
        self.ax.set_ylim([event.ydata - (event.ydata - ylim[0]) * scale,
                          event.ydata + (ylim[1] - event.ydata) * scale])

        self.fig.canvas.draw()

    # -------------------------------
    # PAN (right-click drag)
    # -------------------------------
    def on_click(self, event):
        # Right mouse -> store for panning
        if event.button == 3:
            self.press_event = event
            return

        # Left mouse -> move active point
        if event.button == 1:
            if self.active_point is None:
                print("Select a point with 1–4 first.")
                return

            if event.xdata is None or event.ydata is None:
                return

            self.coords[self.active_point] = [event.xdata, event.ydata]
            print(
                f"Moved point {self.active_point+1} → ({event.xdata:.1f}, {event.ydata:.1f})")

            self.scatter.set_offsets(self.coords)
            self.fig.canvas.draw()

    def on_motion(self, event):
        """Pan the image when right mouse is held."""
        if self.press_event is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        dx = self.press_event.xdata - event.xdata
        dy = self.press_event.ydata - event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.set_xlim(xlim + dx)
        self.ax.set_ylim(ylim + dy)
        self.fig.canvas.draw()

        self.press_event = event

    def on_release(self, event):
        self.press_event = None

    # -------------------------------
    # KEYBOARD
    # -------------------------------
    def on_key(self, event):
        if event.key in ['1', '2', '3', '4']:
            self.active_point = int(event.key) - 1
            print(f"Selected point {self.active_point+1}")

        elif event.key == '5':   # next image
            self.save_current()
            if self.index < len(self.dataset) - 1:
                self.index += 1
                self.show_image()

        elif event.key == '`':   # previous image
            self.save_current()
            if self.index > 0:
                self.index -= 1
                self.show_image()

        elif event.key == 'c':   # toggle faulty
            if self.index in self.faulty:
                self.faulty.remove(self.index)
                print("Unmarked faulty.")
            else:
                self.faulty.add(self.index)
                print("Marked faulty.")
            self.show_image()

        elif event.key == 'o':   # quit
            self.save_current()
            self.finish()

    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    def finish(self):
        output = {
            "coords": self.edited,
            "faulty": sorted(self.faulty),
        }

        with open("annotations.json", "w") as f:
            json.dump(output, f, indent=2)

        print("\nSaved annotations.json")
        print("Faulty images:", sorted(self.faulty))
        plt.close(self.fig)


# Example usage:
ds = ChessDataset.ChessDataset(
    # "/mnt/C/CNN_Dataset/Dataset/",
    "/mnt/D/University/Thesis_Dataset/chessred2k",
    config={
        "img_size": (480, 640)
    },
    force_build_pkl=True
)
annotator = Annotator(ds)
plt.show()
