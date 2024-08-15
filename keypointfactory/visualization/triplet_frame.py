import numpy as np

from . import viz2d
from .local_frame import LocalFrame
from .tools import FormatPrinter


class TripletFrame(LocalFrame):
    default_conf = {
        "default": "keypoints",
        "summary_visible": False,
    }

    def _init_frame(self):
        view0, view1, view2 = self.data["view0"], self.data["view1"], self.data["view2"]
        if self.plot == "color" or self.plot == "color+depth":
            imgs = [
                view0["image"][0].permute(1, 2, 0),
                view1["image"][0].permute(1, 2, 0),
                view2["image"][0].permute(1, 2, 0),
            ]
        elif self.plot == "depth":
            imgs = [view0["depth"][0], view1["depth"][0], view2["depth"][0]]
        else:
            raise ValueError(self.plot)

        imgs = [imgs for _ in self.names]

        fig, axes = viz2d.plot_image_grid(imgs, return_fig=True, titles=None, figs=5)

        [viz2d.add_text(0, n, axes=axes[i]) for i, n in enumerate(self.names)]

        if (
            self.plot == "color+depth"
            and "depth" in view0.keys()
            and view0["depth"] is not None
        ):
            hmaps = [[view0["depth"][0], view1["depth"][0]] for _ in self.names]
            [
                viz2d.plot_heatmaps(hmaps[i], axes=axes[i], cmap="Spectral")
                for i, _ in enumerate(hmaps)
            ]

        if self.summaries is not None:
            formatter = FormatPrinter({np.float32: "%.4f", np.float64: "%.4f"})
            toggle_artists = [
                viz2d.add_text(
                    0,
                    formatter.pformat(self.summaries[n]),
                    axes=axes[i],
                    pos=(0.01, 0.01),
                    va="bottom",
                    backgroundcolor=(0, 0, 0, 0.5),
                    visible=self.conf.summary_visible,
                )
                for i, n in enumerate(self.names)
            ]
        else:
            toggle_artists = []

        return fig, axes, toggle_artists
