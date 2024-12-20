import inspect
import pprint
import sys
import warnings

import matplotlib.pyplot as plt
import torch
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import RadioButtons, Slider

from ..geometry.depth import dense_warp_consistency, simple_project, unproject
from ..geometry.epipolar import T_to_F, generalized_epi_dist
from ..geometry.homography import sym_homography_error, warp_points_torch
from ..visualization.viz2d import (
    cm_ranking,
    cm_RdGn,
    draw_epipolar_line,
    get_line,
    plot_color_line_matches,
    plot_heatmaps,
    plot_keypoints,
    plot_lines,
    plot_matches,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.rcParams["toolbar"] = "toolmanager"


class RadioHideTool(ToolToggleBase):
    """Show lines with a given gid."""

    default_keymap = "R"
    description = "Show by gid"
    default_toggled = False
    radio_group = "default"

    def __init__(
        self, *args, options=[], active=None, callback_fn=None, keymap="R", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.f = 1.0
        self.options = options
        self.callback_fn = callback_fn
        self.active = self.options.index(active) if active else 0
        self.default_keymap = keymap

        self.enabled = self.default_toggled

    def build_radios(self):
        w = 0.2
        self.radios_ax = self.figure.add_axes([1.0 - w, 0.7, w, 0.2], zorder=1)
        # self.radios_ax = self.figure.add_axes([0.5-w/2, 1.0-0.2, w, 0.2], zorder=1)
        self.radios = RadioButtons(self.radios_ax, self.options, active=self.active)
        self.radios.on_clicked(self.on_radio_clicked)

    def enable(self, *args):
        size = self.figure.get_size_inches()
        size[0] *= self.f
        self.build_radios()
        self.figure.canvas.draw_idle()
        self.enabled = True

    def disable(self, *args):
        size = self.figure.get_size_inches()
        size[0] /= self.f
        self.radios_ax.remove()
        self.radios = None
        self.figure.canvas.draw_idle()
        self.enabled = False

    def on_radio_clicked(self, value):
        self.active = self.options.index(value)
        enabled = self.enabled
        if enabled:
            self.disable()
        if self.callback_fn is not None:
            self.callback_fn(value)
        if enabled:
            self.enable()


class ToggleTool(ToolToggleBase):
    """Show lines with a given gid."""

    default_keymap = "t"
    description = "Show by gid"

    def __init__(self, *args, callback_fn=None, keymap="t", **kwargs):
        super().__init__(*args, **kwargs)
        self.f = 1.0
        self.callback_fn = callback_fn
        self.default_keymap = keymap
        self.enabled = self.default_toggled

    def enable(self, *args):
        self.callback_fn(True)

    def disable(self, *args):
        self.callback_fn(False)


class FormatPrinter(pprint.PrettyPrinter):
    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)


def add_whitespace_left(fig, factor):
    w, h = fig.get_size_inches()
    left = fig.subplotpars.left
    fig.set_size_inches([w * (1 + factor), h])
    fig.subplots_adjust(left=(factor + left) / (1 + factor))


def add_whitespace_bottom(fig, factor):
    w, h = fig.get_size_inches()
    b = fig.subplotpars.bottom
    fig.set_size_inches([w, h * (1 + factor)])
    fig.subplots_adjust(bottom=(factor + b) / (1 + factor))
    fig.canvas.draw_idle()


class KeypointPlot:
    plot_name = "keypoints"
    required_keys = ["keypoints0", "keypoints1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            kpts = [pred["keypoints0"][0], pred["keypoints1"][0]]
            if "keypoints2" in pred and len(axes[i]) > 2:
                kpts.append(pred["keypoints2"][0])
            plot_keypoints(kpts, axes=axes[i])


class InteractiveHomographyKeypointPlot:
    plot_name = "interactive_homography_keypoints"
    required_keys = ["keypoints0", "keypoints1", "H_0to1"]

    colors = ["blue", "red", "yellow"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.axes = axes
        self.data = data
        self.preds = preds
        for i, name in enumerate(preds):
            pred = preds[name]
            kpts = [pred["keypoints0"][0], pred["keypoints1"][0]]
            if "keypoints2" in pred and len(axes[i]) > 2:
                kpts.append(pred["keypoints2"][0])
            plot_keypoints(kpts, axes=axes[i], pickable=True)

    def click_artist(self, event):
        art = event.artist
        if isinstance(art, PathCollection):
            for i in range(len(self.axes)):
                if art.axes in self.axes[i]:
                    break
            axes_index = self.axes[i].tolist().index(event.artist.axes)
            ind = event.ind
            xy = self.preds[list(self.preds.keys())[i]][f"keypoints{axes_index}"][0][
                ind, :
            ]
            self._plot_reprojection(xy, i, axes_index)
            handles = [
                Line2D([0], [0], marker=".", markersize=3, color=color, linestyle=None)
                for color in self.colors
            ]
            self.fig.legend(handles, ["view0", "view1", "view2"], loc="upper right")

    def _plot_reprojection(self, xy, row_idx, index):
        if index == 0:
            index_tuple = (0, 1, "H_0to1", False, 2, "H_0to2", False)
        elif index == 1:
            index_tuple = (1, 0, "H_0to1", True, 2, "H_1to2", False)
        else:
            index_tuple = (2, 0, "H_0to2", True, 1, "H_1to2", True)

        if f"view{index_tuple[1]}" in self.data.keys():
            H = self.data[index_tuple[2]][0]
            xy_r = warp_points_torch(xy, H, inverse=index_tuple[3])
            image_width, image_height = self.data[f"view{index_tuple[1]}"][
                "image_size"
            ][0]
            xy_r = xy_r[
                (xy_r[:, 0] >= 0 & (xy_r[:, 0] < image_width))
                & (xy_r[:, 1] >= 0 & (xy_r[:, 1] < image_height))
            ]
            plot_keypoints(
                [xy_r],
                axes=[self.axes[row_idx][index_tuple[1]]],
                colors=self.colors[index_tuple[0]],
                pickable=False,
            )

        if f"view{index_tuple[4]}" in self.data.keys():
            H = self.data[index_tuple[5]][0]
            xy_r = warp_points_torch(xy, H, inverse=index_tuple[6])
            image_width, image_height = self.data[f"view{index_tuple[4]}"][
                "image_size"
            ][0]
            xy_r = xy_r[
                (xy_r[:, 0] >= 0 & (xy_r[:, 0] < image_width))
                & (xy_r[:, 1] >= 0 & (xy_r[:, 1] < image_height))
            ]
            plot_keypoints(
                [xy_r],
                axes=[self.axes[row_idx][index_tuple[4]]],
                colors=self.colors[index_tuple[0]],
                pickable=False,
            )


class InteractiveEpipolarKeypointPlot:
    plot_name = "interactive_epipolar_keypoints"
    required_keys = ["keypoints0", "keypoints1", "T_0to1", "view0", "view1"]

    line_colors = ["tab:blue", "tab:red", "tab:olive"]
    kpt_colors = ["tab:purple", "tab:orange", "tab:cyan"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.axes = axes
        self.data = data
        self.preds = preds
        for i, name in enumerate(preds):
            pred = preds[name]
            kpts = [pred["keypoints0"][0], pred["keypoints1"][0]]
            if "keypoints2" in pred and len(axes[i]) > 2:
                kpts.append(pred["keypoints2"][0])
            plot_keypoints(kpts, axes=axes[i], pickable=True)

        self.F = {
            "F_0to1": T_to_F(
                data["view0"]["camera"][0],
                data["view1"]["camera"][0],
                data["T_0to1"][0],
            )
        }
        if "view2" in data:
            self.F["F_0to2"] = T_to_F(
                data["view0"]["camera"][0],
                data["view2"]["camera"][0],
                data["T_0to2"][0],
            )
            self.F["F_1to2"] = T_to_F(
                data["view1"]["camera"][0],
                data["view2"]["camera"][0],
                data["T_1to2"][0],
            )

    def click_artist(self, event):
        art = event.artist
        if isinstance(art, PathCollection):
            for i in range(len(self.axes)):
                if art.axes in self.axes[i]:
                    break
            axes_index = self.axes[i].tolist().index(event.artist.axes)
            ind = [event.ind[0]]
            xy = self.preds[list(self.preds.keys())[i]][f"keypoints{axes_index}"][0][
                ind, :
            ]
            self._plot_epipolar(xy, i, axes_index)
            if "depth" in self.data["view0"].keys():
                self._plot_reprojection(xy, i, axes_index)
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker=".",
                    markersize=3,
                    color=l_color,
                    markeredgecolor=k_color,
                    linestyle="dashed",
                )
                for l_color, k_color in zip(self.line_colors, self.kpt_colors)
            ]
            self.fig.legend(handles, ["view0", "view1", "view2"], loc="upper right")

    def _plot_epipolar(self, xy, row_idx, index):
        if index == 0:
            index_tuple = (0, 1, "F_0to1", False, 2, "F_0to2", False)
        elif index == 1:
            index_tuple = (1, 0, "F_0to1", True, 2, "F_1to2", False)
        else:
            index_tuple = (2, 0, "F_0to2", True, 1, "F_1to2", True)

        for j in range(xy.shape[0]):
            if f"view{index_tuple[1]}" in self.data.keys():
                F = (
                    self.F[index_tuple[2]]
                    if not index_tuple[3]
                    else self.F[index_tuple[2]].transpose(0, 1)
                )
                line = get_line(F, xy[j, :])[:, 0]
                draw_epipolar_line(
                    line,
                    self.axes[row_idx][index_tuple[1]],
                    self.data[f"view{index_tuple[1]}"]["image_size"][0]
                    .flip(dims=(0,))
                    .numpy(),
                    color=self.line_colors[index_tuple[0]],
                    pickable=False,
                )

            if f"view{index_tuple[4]}" in self.data.keys():
                F = (
                    self.F[index_tuple[5]]
                    if not index_tuple[6]
                    else self.F[index_tuple[5]].transpose(0, 1)
                )
                line = get_line(F, xy[j, :])[:, 0]
                draw_epipolar_line(
                    line,
                    self.axes[row_idx][index_tuple[4]],
                    self.data[f"view{index_tuple[4]}"]["image_size"][0]
                    .flip(dims=(0,))
                    .numpy(),
                    color=self.line_colors[index_tuple[0]],
                    pickable=False,
                )

    def _plot_reprojection(self, xy, row_idx, index):
        if index == 0:
            index_tuple = (0, 1, 2)
        elif index == 1:
            index_tuple = (1, 0, 2)
        else:
            index_tuple = (2, 0, 1)

        depth = self.data[f"view{index_tuple[0]}"]["depth"]
        cam0 = self.data[f"view{index_tuple[0]}"]["camera"]
        T0 = self.data[f"view{index_tuple[0]}"]["T_w2cam"]

        if f"view{index_tuple[1]}" in self.data.keys():
            cam1 = self.data[f"view{index_tuple[1]}"]["camera"]
            T1 = self.data[f"view{index_tuple[1]}"]["T_w2cam"]
            xy_r = simple_project(
                unproject(xy.unsqueeze(0), depth, cam0, T0), cam1, T1
            )[0]
            image_width, image_height = self.data[f"view{index_tuple[1]}"][
                "image_size"
            ][0]
            xy_r = xy_r[
                (xy_r[:, 0] >= 0 & (xy_r[:, 0] < image_width))
                & (xy_r[:, 1] >= 0 & (xy_r[:, 1] < image_height))
            ]
            plot_keypoints(
                [xy_r],
                ps=20,
                axes=[self.axes[row_idx][index_tuple[1]]],
                colors=self.kpt_colors[index_tuple[0]],
                pickable=False,
            )

        if f"view{index_tuple[2]}" in self.data.keys():
            cam1 = self.data[f"view{index_tuple[2]}"]["camera"]
            T1 = self.data[f"view{index_tuple[2]}"]["T_w2cam"]
            xy_r = simple_project(
                unproject(xy.unsqueeze(0), depth, cam0, T0), cam1, T1
            )[0]
            image_width, image_height = self.data[f"view{index_tuple[2]}"][
                "image_size"
            ][0]
            xy_r = xy_r[
                (xy_r[:, 0] >= 0 & (xy_r[:, 0] < image_width))
                & (xy_r[:, 1] >= 0 & (xy_r[:, 1] < image_height))
            ]
            plot_keypoints(
                [xy_r],
                ps=20,
                axes=[self.axes[row_idx][index_tuple[2]]],
                colors=self.kpt_colors[index_tuple[0]],
                pickable=False,
            )


class DenseConsistencyPlot:
    plot_name = "dense_consistency"
    required_keys = ["view0", "view1", "T_0to1"]

    def __init__(self, fig, axes, data, preds):
        depth0 = data["view0"]["depth"][0]
        depth1 = data["view1"]["depth"][0]
        camera0 = data["view0"]["camera"][0]
        camera1 = data["view1"]["camera"][0]
        T = data["T_0to1"][0]

        dkpts0, valid0 = dense_warp_consistency(
            depth1, depth0, T.inv(), camera1, camera0
        )
        dkpts1, valid0 = dense_warp_consistency(depth0, depth1, T, camera0, camera1)

        [
            plot_keypoints(
                [dkpts0[0].flatten(0, 1), dkpts1[0].flatten(0, 1)],
                ps=1,
                a=0.5,
                axes=axes[i],
            )
            for i in range(len(preds))
        ]


class LinePlot:
    plot_name = "lines"
    required_keys = ["lines0", "lines1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            plot_lines([pred["lines0"][0], pred["lines1"][0]])


class KeypointRankingPlot:
    plot_name = "keypoint_ranking"
    required_keys = ["keypoints0", "keypoints1", "keypoint_scores0", "keypoint_scores1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]

            kpts = [pred["keypoints0"][0], pred["keypoints1"][0]]
            scores = [pred["keypoint_scores0"][0], pred["keypoint_scores1"][0]]

            if "keypoints2" in pred:
                kpts.append(pred["keypoints2"][0])
                scores.append(pred["keypoint_scores2"][0])

            plot_keypoints(kpts, axes=axes[i], colors=[cm_ranking(sc) for sc in scores])


class KeypointScoresPlot:
    plot_name = "keypoint_scores"
    required_keys = ["keypoints0", "keypoints1", "keypoint_scores0", "keypoint_scores1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            kpts = [pred["keypoints0"][0], pred["keypoints1"][0]]
            scores = [pred["keypoint_scores0"][0], pred["keypoint_scores1"][0]]

            if "keypoints2" in pred:
                kpts.append(pred["keypoints2"][0])
                scores.append(pred["keypoint_scores2"][0])

            plot_keypoints(kpts, axes=axes[i], colors=[cm_RdGn(sc) for sc in scores])


class HeatmapPlot:
    plot_name = "heatmaps"
    required_keys = ["heatmap0", "heatmap1"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        for i, name in enumerate(preds):
            pred = preds[name]
            heatmaps = [pred["heatmap0"][0, 0], pred["heatmap1"][0, 0]]
            if "heatmap2" in pred:
                heatmaps.append(pred["heatmap2"][0, 0])
            heatmaps = [torch.sigmoid(h) if h.min() < 0.0 else h for h in heatmaps]
            self.artists += plot_heatmaps(heatmaps, axes=axes[i], cmap="rainbow")

    def clear(self):
        for x in self.artists:
            x.remove()


class ImagePlot:
    plot_name = "images"
    required_keys = ["view0", "view1"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        if "depth" in data["view0"].keys():
            for i, _ in enumerate(preds):
                depths = [
                    data["view0"]["depth"][0],
                    data["view1"]["depth"][0],
                ]

                if "view2" in data:
                    depths.append(data["view2"]["depth"][0])

                self.artists += plot_heatmaps(depths, axes=axes[i], cmap="rainbow")

    def clear(self):
        for x in self.artists:
            x.remove()


class MatchesPlot:
    plot_name = "matches"
    required_keys = ["keypoints0", "keypoints1", "matches0", "matching_scores0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            mscores = pred["matching_scores0"][0][valid]
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(mscores).tolist(),
                axes=axes[i],
                labels=mscores,
                lw=0.5,
            )


class LineMatchesPlot:
    plot_name = "line_matches"
    required_keys = ["lines0", "lines1", "line_matches0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            lines0, lines1 = pred["lines0"][0], pred["lines1"][0]
            m0 = pred["line_matches0"][0]
            valid = m0 > -1
            m_lines0 = lines0[valid]
            m_lines1 = lines1[m0[valid]]
            plot_color_line_matches([m_lines0, m_lines1])


class GtMatchesPlot:
    plot_name = "gt_matches"
    required_keys = ["keypoints0", "keypoints1", "matches0", "gt_matches0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            gtm0 = pred["gt_matches0"][0]
            valid = (m0 > -1) & (gtm0 >= -1)
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            correct = gtm0[valid] == m0[valid]
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(correct).tolist(),
                axes=axes[i],
                labels=correct,
                lw=0.5,
            )


class GtLineMatchesPlot:
    plot_name = "gt_line_matches"
    required_keys = ["lines0", "lines1", "line_matches0", "line_gt_matches0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            lines0, lines1 = pred["lines0"][0], pred["lines1"][0]
            m0 = pred["line_matches0"][0]
            gtm0 = pred["gt_line_matches0"][0]
            valid = (m0 > -1) & (gtm0 >= -1)
            m_lines0 = lines0[valid]
            m_lines1 = lines1[m0[valid]]
            plot_color_line_matches([m_lines0, m_lines1])


class HomographyMatchesPlot:
    plot_name = "homography"
    required_keys = ["keypoints0", "keypoints1", "matches0", "H_0to1"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        add_whitespace_bottom(fig, 0.1)

        self.range_ax = fig.add_axes([0.3, 0.02, 0.4, 0.06])
        self.range = Slider(
            self.range_ax,
            label="Homography Error",
            valmin=0,
            valmax=5,
            valinit=3.0,
            valstep=1.0,
        )
        self.range.on_changed(self.color_matches)

        for i, name in enumerate(preds):
            pred = preds[name]
            plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            errors = sym_homography_error(kpm0, kpm1, data["H_0to1"][0])
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(errors < self.range.val).tolist(),
                axes=axes[i],
                labels=errors.numpy(),
                lw=0.5,
            )

    def clear(self):
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(w, h / 1.1)
        self.fig.subplots_adjust(**self.sbpars)
        self.range_ax.remove()

    def color_matches(self, args):
        for line in self.fig.artists:
            label = line.get_label()
            line.set_color(cm_RdGn([float(label) < args])[0])


class EpipolarMatchesPlot:
    plot_name = "epipolar_matches"
    required_keys = ["keypoints0", "keypoints1", "matches0", "T_0to1", "view0", "view1"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.axes = axes
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        add_whitespace_bottom(fig, 0.1)

        self.range_ax = fig.add_axes([0.3, 0.02, 0.4, 0.06])
        self.range = Slider(
            self.range_ax,
            label="Epipolar Error [px]",
            valmin=0,
            valmax=5,
            valinit=3.0,
            valstep=1.0,
        )
        self.range.on_changed(self.color_matches)

        camera0 = data["view0"]["camera"][0]
        camera1 = data["view1"]["camera"][0]
        T_0to1 = data["T_0to1"][0]

        for i, name in enumerate(preds):
            pred = preds[name]
            plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]

            errors = generalized_epi_dist(
                kpm0,
                kpm1,
                camera0,
                camera1,
                T_0to1,
                all=False,
                essential=False,
            )
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(errors < self.range.val).tolist(),
                axes=axes[i],
                labels=errors.numpy(),
                lw=0.5,
            )

        self.F = T_to_F(camera0, camera1, T_0to1)

    def clear(self):
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(w, h / 1.1)
        self.fig.subplots_adjust(**self.sbpars)
        self.range_ax.remove()

    def color_matches(self, args):
        for art in self.fig.artists:
            label = art.get_label()
            if label is not None:
                art.set_color(cm_RdGn([float(label) < args])[0])

    def click_artist(self, event):
        art = event.artist
        if art.get_label() is not None:
            if hasattr(art, "epilines"):
                [
                    x.set_visible(not x.get_visible())
                    for x in art.epilines
                    if x is not None
                ]
            else:
                xy1 = art.xy1
                xy2 = art.xy2
                line0 = get_line(self.F.transpose(0, 1), xy2)[:, 0]
                line1 = get_line(self.F, xy1)[:, 0]
                art.epilines = [
                    draw_epipolar_line(line0, art.axesA),
                    draw_epipolar_line(line1, art.axesB),
                ]


__plot_dict__ = {
    obj.plot_name: obj
    for _, obj in inspect.getmembers(sys.modules[__name__], predicate=inspect.isclass)
    if hasattr(obj, "plot_name")
}
