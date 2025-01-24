import datasets
from datasets.config import importlib_metadata, version
import evaluate

import numpy as np
from sklearn import metrics
import random

_CITATION = """\
@article{xxx
}
"""

_DESCRIPTION = """\
Classification metrics for DTI.
"""

_KWARGS_DESCRIPTION = """
No args.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class molnet_regression(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("float", id="sequence"),
                        "references": datasets.Value("float", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://xxx.com"],
            reference_urls=[
                "https://xxx.com"
            ],
        )

    def _compute(self, predictions, references, tsv_path="tmp.tsv"):
        y_pred = predictions
        y_true = references

        # Calculate metrics: RMSE, MAE
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mae = metrics.mean_absolute_error(y_true, y_pred)

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("ground truth\toutput\n")
            for gt, out in zip(y_true, y_pred):
                f.write(str(gt) + "\t" + str(out) + "\n")

        return {
            "rmse": rmse,
            "mae": mae,
        }
