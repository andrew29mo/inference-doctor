"""Finite classification dataset execution shared by the production worker.

All rows are used; the last batch is short rather than padded. Timers surround
only synchronous runtime calls; validation, classification and audit hashes are
outside those timers and inside the separate pass-wall timer.
"""

import hashlib
import math
import time

import numpy as np

from .validation import DiagnosticError, load_arrays


def int_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()


class ClassificationDataset:
    def __init__(self, inputs, reference, labels_path, batch_size, output_name):
        labels = load_arrays(labels_path)
        if set(labels) != {"labels", "sample_ids", "classes"}:
            raise DiagnosticError("dataset_labels", "Labels NPZ must contain labels, sample_ids and classes.")
        self.labels, self.ids, self.classes = (labels[k] for k in ("labels", "sample_ids", "classes"))
        if any(a.ndim != 1 or a.dtype != np.dtype("int64") for a in labels.values()):
            raise DiagnosticError("dataset_labels", "Label, ID and class tensors must be 1D int64.")
        self.rows = len(self.labels)
        self.batch_size = batch_size
        self.inputs, self.reference, self.output_name = inputs, reference, output_name
        if self.rows == 0 or len(self.ids) != self.rows or len(np.unique(self.ids)) != self.rows:
            raise DiagnosticError("dataset_ids", "Dataset row IDs must be nonempty, unique and match the label count.")
        if len(self.classes) < 2 or len(np.unique(self.classes)) != len(self.classes) or not np.isin(self.labels, self.classes).all():
            raise DiagnosticError("dataset_classes", "Classes must be unique and include every label.")
        if batch_size <= 0 or not inputs or reference is None or output_name not in reference:
            raise DiagnosticError("dataset_config", "Dataset mode requires inputs, references, a positive batch size and a classification output.")
        if any(a.ndim < 1 or len(a) != self.rows for a in [*inputs.values(), *reference.values()]):
            raise DiagnosticError("dataset_rows", "Every input/reference must have the same leading dataset dimension.")
        scores = reference[output_name]
        if scores.ndim != 2 or scores.shape[1] != len(self.classes) or scores.dtype.kind != "f":
            raise DiagnosticError("dataset_output", "Classification reference must be a floating [rows, classes] tensor.")
        self.reference_predictions = self.classes[scores.argmax(axis=1)]
        self.batches = math.ceil(self.rows / self.batch_size)
        self.expected_correct = int(np.sum(self.reference_predictions == self.labels))

    def slices(self):
        for start in range(0, self.rows, self.batch_size):
            yield slice(start, min(start + self.batch_size, self.rows))

    def feed(self, subset):
        return {name: values[subset] for name, values in self.inputs.items()}

    def expected(self, subset):
        return {name: values[subset] for name, values in self.reference.items()}

    def metadata(self):
        return {"rows": self.rows, "batch_size": self.batch_size, "batches_per_pass": self.batches,
                "last_batch_rows": self.rows - (self.batches - 1) * self.batch_size,
                "classification_output": self.output_name, "classes": self.classes.tolist(),
                "reference_correct": self.expected_correct, "reference_accuracy": self.expected_correct / self.rows,
                "sample_ids_sha256": int_hash(self.ids), "reference_predictions_sha256": int_hash(self.reference_predictions)}

    def run_pass(self, session, names, runtime_call, validate, timed):
        start_wall = time.perf_counter_ns()
        samples, sample_rows, predictions, observed_ids = [], [], [], []
        for subset in self.slices():
            feed = self.feed(subset)
            expected = self.expected(subset)
            start = time.perf_counter_ns()
            outputs = runtime_call("dataset_inference", lambda: session.run(None, feed))
            elapsed = (time.perf_counter_ns() - start) / 1e6
            validate(outputs, expected)
            scores = outputs[names.index(self.output_name)]
            if scores.ndim != 2 or scores.shape != (subset.stop - subset.start, len(self.classes)):
                raise DiagnosticError("dataset_output", "Runtime classification shape differs from the actual batch.")
            predicted = self.classes[scores.argmax(axis=1)]
            if not np.array_equal(predicted, self.reference_predictions[subset]):
                raise DiagnosticError("output_classification", "A predicted class differs from the independent reference.")
            predictions.append(predicted)
            observed_ids.append(self.ids[subset])
            if timed:
                samples.append(elapsed)
                sample_rows.append(len(predicted))
            del outputs, scores
        predicted = np.concatenate(predictions)
        ids = np.concatenate(observed_ids)
        if not np.array_equal(ids, self.ids) or len(predicted) != self.rows:
            raise DiagnosticError("dataset_coverage", "A dataset pass skipped, repeated or reordered rows.")
        correct = int(np.sum(predicted == self.labels))
        evidence = {"valid_rows": len(predicted), "calls": len(observed_ids), "correct": correct,
                    "accuracy": correct / self.rows, "sample_ids_sha256": int_hash(ids),
                    "predictions_sha256": int_hash(predicted), "inference_ms": sum(samples) if timed else None,
                    "validated_wall_ms": (time.perf_counter_ns() - start_wall) / 1e6}
        return evidence, samples, sample_rows
