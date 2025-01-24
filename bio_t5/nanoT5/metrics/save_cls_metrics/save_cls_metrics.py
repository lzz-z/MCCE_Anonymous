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
class Save_classification(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
            ],
            codebase_urls=["https://xxx.com"],
            reference_urls=[
                "https://xxx.com"
            ],
        )

    def _compute(self, predictions, references, tsv_path="tmp.tsv"):
        gts = [ref[0] for ref in references]
        input_mols = [ref[1] for ref in references]

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("molecule\toutput\tgt\n")
            for it, out, gt in zip(input_mols, predictions, gts):
                f.write(str(it) + "\t" + str(out) + "\t" + str(gt) + "\n")

        return {}
