# Third-party data and dependencies

The real-model example uses scikit-learn's bundled **Digits** dataset: 1,797 images, each 8×8 pixels, from the UCI Optical Recognition of Handwritten Digits dataset by E. Alpaydin and C. Kaynak (1998).

- [UCI dataset and license](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [Dataset DOI](https://doi.org/10.24432/C50P49)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- [scikit-learn bundled subset documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.datasets.load_digits.html)

Changes for this experiment: a fixed stratified split of the bundled subset, normalization by 16, numeric labels/sample IDs, a trained classifier, and generated reference predictions/reports. This is a new local split, not the original UCI train/test protocol. Dataset-derived fixture values and reports retain the source-data attribution and CC BY 4.0 terms where applicable. The generated binary fixtures are ignored; retained numeric evidence is intentional.

scikit-learn and SciPy use BSD licenses; NumPy, ONNX, ONNX Runtime and the other dependencies retain their respective upstream licenses. They are installed as dependencies; their source is not vendored here.

The repository's original code, documentation and trained weight state are licensed under the [MIT License](LICENSE), Copyright (c) 2026 Andrew Yuan. This grant does not relicense the Digits source data, dataset-derived fixture values or other third-party materials, and does not imply endorsement by the dataset authors. Preserve the dataset attribution, CC BY 4.0 link and change description above when redistributing dataset-derived materials. Dependencies retain their upstream licenses.
