from .tools import RadioHideTool, ToggleTool, __plot_dict__

class LocalFrame:
    default_conf = {
        "default": "keypoints",
        "summary_visible": False,
        "points_visible": False,
    }

    plot_dict = __plot_dict__

    childs = []

    event_to_image = [None, "color", "depth", "color+depth"]

    def __init__(self, conf, data, preds, title=None, event=1, summaries=None):
        self.conf = conf
        self.data = data
        self.preds = preds
        self.names = list(preds.keys())
        self.plot = self.event_to_image[event]
        self.summaries = summaries
        self.fig, self.axes, self.summary_arts = self.init_frame()
        self.fig.canvas.mpl_connect("pick_event", self.click_artist)
        if title is not None:
            self.fig.canvas.manager.set_window_title(title)

        keys = None
        for _, pred in preds.items():
            if keys is None:
                keys = set(pred.keys())
            else:
                keys = keys.intersection(pred.keys())
        keys = keys.union(data.keys())

        self.options = [
            k for k, v in self.plot_dict.items() if set(v.required_keys).issubset(keys)
        ]
        self.handle = None
        self.radios = self.fig.canvas.manager.toolmanager.add_tool(
            "switch plot",
            RadioHideTool,
            options=self.options,
            callback_fn=self.draw,
            active=conf.default,
            keymap="R",
        )

        self.toggle_summary = self.fig.canvas.manager.toolmanager.add_tool(
            "toggle summary",
            ToggleTool,
            toggled=self.conf.summary_visible,
            callback_fn=self.set_summary_visible,
            keymap="t",
        )
        
        if self.fig.canvas.manager.toolbar is not None:
            self.fig.canvas.manager.toolbar.add_tool("switch plot", "navigation")
        
        self.draw(conf.default)

    def init_frame(self):
        """initialize frame"""
        return self._init_frame()

    def _init_frame(self):
        """initialize frame"""
        raise NotImplementedError
    
    def click_artist(self, event):
        self._click_artist(event)
        if hasattr(self.handle, "click_artist"):
            self.handle.click_artist(event)
        self.fig.canvas.draw_idle()

    def _click_artist(self, event):
        pass

    def draw(self, value):
        """redraw content in frame"""
        self.clear()
        self.conf.default = value
        self.handle = self.plot_dict[value](self.fig, self.axes, self.data, self.preds)
        return self.handle
    
    def clear(self):
        if self.handle is not None:
            try:
                self.handle.clear()
            except AttributeError:
                pass
        self.handle = None
        for row in self.axes:
            for ax in row:
                [li.remove() for li in ax.lines]
                [c.remove() for c in ax.collections]
        self.fig.artists.clear()
        self.fig.canvas.draw_idle()
        self.handle = None

    def set_summary_visible(self, visible):
        self.conf.summary_visible = visible
        [s.set_visible(visible) for s in self.summary_arts]
        self.fig.canvas.draw_idle()

    def set_points_visible(self, visible):
        self.conf.points_visible = visible
        self.draw(self.conf.default)
        self.fig.canvas.draw_idle()

    