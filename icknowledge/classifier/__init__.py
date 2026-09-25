"""Classifier training + labeling.

Group-blind logistic regression trained on the descendant features (A excluded),
with a thresholded label whose noise scale is calibrated so h is non-trivial but
not ~perfect. See the `training` and `labeling` modules.
"""

from icknowledge.classifier.labeling import (
    LabelSpec,
    compute_signal,
    make_labels,
)
from icknowledge.classifier.training import (
    ClassifierResult,
    build_classifier,
    classifier_provenance,
    format_report,
)

__all__ = [
    "LabelSpec",
    "compute_signal",
    "make_labels",
    "ClassifierResult",
    "build_classifier",
    "classifier_provenance",
    "format_report",
]
