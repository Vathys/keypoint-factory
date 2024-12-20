import pandas as pd
from omegaconf import OmegaConf


def load_eval(dir):
    results = pd.read_hdf(str(dir / "results.h5"), key="results", mode="r")
    summaries = pd.read_json(str(dir / "summaries.json"), orient="records")
    return summaries, results


def save_eval(dir, summaries, figures, results):
    results.to_hdf(
        str(dir / "results.h5"), key="results", mode="w", format="fixed", index=False
    )
    summaries.to_json(str(dir / "summaries.json"), orient="records", indent=4, mode="w")
    for fig_name, fig in figures.items():
        fig.savefig(dir / f"{fig_name}.png")


def exists_eval(dir):
    return (dir / "results.h5").exists() and (dir / "summaries.json").exists()


class EvalPipeline:
    default_conf = {}

    export_keys = []
    optional_export_keys = []

    def __init__(self, conf):
        """Assumes"""
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self._init(self.conf)

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        raise NotImplementedError

    def get_predictions(
        self, experiment_dir, model=None, overwrite=False, get_last=False
    ):
        """Export a prediction file for each eval datapoint"""
        raise NotImplementedError

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        raise NotImplementedError

    def run(
        self,
        experiment_dir,
        model=None,
        overwrite=False,
        overwrite_eval=False,
        get_last=False,
    ):
        """Run export+eval loop"""
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite, get_last=get_last
        )

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(self.get_dataloader(), pred_file)
            save_eval(experiment_dir, s, f, r)
        s, r = load_eval(experiment_dir)
        return s, f, r

    def save_conf(self, experiment_dir, overwrite=False, overwrite_eval=False):
        # store config
        conf_output_path = experiment_dir / "conf.yaml"
        if conf_output_path.exists():
            saved_conf = OmegaConf.load(conf_output_path)
            if (saved_conf.data != self.conf.data) or (
                saved_conf.model != self.conf.model
            ):
                assert (
                    overwrite
                ), "configs changed, add --overwrite to rerun experiment with new conf"
            if saved_conf.eval != self.conf.eval:
                assert (
                    overwrite or overwrite_eval
                ), "eval configs changed, add --overwrite_eval to rerun evaluation"
        OmegaConf.save(self.conf, experiment_dir / "conf.yaml")
