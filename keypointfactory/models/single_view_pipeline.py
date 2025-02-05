from keypointfactory.models.two_view_pipeline import TwoViewPipeline


class SingleViewPipeline(TwoViewPipeline):
    default_conf = {**TwoViewPipeline.default_conf}
    required_data_keys = []
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _forward(self, data):
        pred = self.extract_view(data, "0")

        return pred

    def _pre_loss_callback(self, seed, epoch):
        super()._pre_loss_callback(seed, epoch)

    def _post_loss_callback(self, seed, epoch):
        super()._post_loss_callback(seed, epoch)

    def _detach_grad_filter(self, key):
        return super()._detach_grad_filter(key)

    def loss(self, pred, data):
        losses = {}
        metrics = {}

        # get labels
        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, data)
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}

        return losses_, metrics_
