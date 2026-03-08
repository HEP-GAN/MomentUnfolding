# MomentUnfolding

Software for moment extraction using an unfolding protocol without binning, based on a GAN-based adversarial learning approach for high-energy physics applications.

This repository accompanies the paper:

> **Moment Extraction Using an Unfolding Protocol Without Binning**  
> Krish Desai, Benjamin Nachman, Jesse Thaler  
> *Physical Review D* **110**, 116013 (2024) — [10.1103/PhysRevD.110.116013](https://doi.org/10.1103/PhysRevD.110.116013)  
> arXiv: [2407.11284 \[hep-ph\]](https://arxiv.org/abs/2407.11284)

---

## Overview

In particle physics experiments, detector effects smear and distort measured observables. Traditional unfolding methods recover the underlying (particle-level) distributions by discretizing data into bins and inverting a response matrix. This approach introduces binning artifacts and information loss.

**MomentUnfolding** learns an optimal event reweighting function $w(x) = e^{\lambda_0 x + \lambda_1 x^2}$ using a GAN-based adversarial framework — no binning required. The weights are tuned so that moments of the reweighted generator-level distribution match the true particle-level moments:

$$\mathbb{E}[x^k \cdot w(x)] \approx \mathbb{E}_\text{truth}[x^k]$$

Key features:
- **Unbinned**: preserves all information in the data
- **Flexible**: works with any continuous observable
- **GAN-based**: adversarial training for robust convergence
- **Extensible**: supports 2D unfolding with momentum-dependent weights

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- Jupyter

Optional (for some notebooks):
- SciPy
- tabulate

Install all dependencies:

```bash
pip install tensorflow numpy scikit-learn matplotlib scipy jupyter tabulate
```

---

## Getting Started

Clone the repository and launch Jupyter:

```bash
git clone https://github.com/HEP-GAN/MomentUnfolding.git
cd MomentUnfolding
jupyter notebook
```

**Start with `GaussianExample.ipynb`** — it requires no external data and runs in about a minute, demonstrating the full unfolding pipeline on a simple Gaussian toy model.

---

## Notebooks

| Notebook | Description | External Data Required |
|---|---|---|
| [`GaussianExample.ipynb`](GaussianExample.ipynb) | Simplest demonstration using synthetic Gaussian data | No |
| [`JetExample.ipynb`](JetExample.ipynb) | Unfolding real LHC jet observables (charge, mass, width, radius, N-subjettiness) | Yes (`npfiles/rawdata.npz`) |
| [`BootstrapErrors.ipynb`](BootstrapErrors.ipynb) | Uncertainty estimation via bootstrap resampling (~500 iterations) | Yes |
| [`2DLoss.ipynb`](2DLoss.ipynb) | Visualization of the loss landscape in $\lambda_0$–$\lambda_1$ parameter space | Yes |
| [`MethodComparison.ipynb`](MethodComparison.ipynb) | Comparison against OmniFold, IBU, and binned methods | Yes |
| [`MomentumDependence.ipynb`](MomentumDependence.ipynb) | 2D unfolding with $p_T$-dependent reweighting | Yes |

The external dataset `npfiles/rawdata.npz` (containing simulated LHC jet events) is not included in this repository. Please contact the authors for access.

---

## Method

The core of the method is a custom Keras layer (`MyLayer`) that parameterizes the reweighting function. A discriminator network is trained adversarially to distinguish truth-level events from reweighted generator-level events. When training converges, the discriminator can no longer distinguish the two distributions, meaning the weighted generator moments equal the truth moments.

**Architecture:**
- *Generator layer*: $w(x) = e^{\lambda_0 x + \lambda_1 x^2}$ (2 learnable parameters for 1D; 4 for 2D)
- *Discriminator*: three Dense layers of 50 ReLU units followed by a sigmoid output

**Training loop (each batch):**
1. Compute event weights $w$ from the generator layer
2. Train the discriminator using weighted binary cross-entropy
3. Train the generator adversarially to minimize the discriminator's ability to separate the distributions

The best $\lambda$ values are selected by minimising the moment-recovery error over training checkpoints.

---

## Citation

If you use this software, please cite both the paper and the software:

```bibtex
@article{desai2024moment,
  author       = {Desai, Krish and Nachman, Benjamin and Thaler, Jesse},
  title        = {Moment Extraction Using an Unfolding Protocol Without Binning},
  journaltitle = {Physical Review D},
  volume       = {110},
  number       = {11},
  eid          = {116013},
  date         = {2024-12-13},
  publisher    = {American Physical Society},
  doi          = {10.1103/PhysRevD.110.116013},
  eprint       = {2407.11284},
  eprintclass  = {hep-ph},
}
```

See also [`CITATION.cff`](CITATION.cff) and [`CITATION.bib`](CITATION.bib) for machine-readable citation metadata.

---

## License

This project is licensed under the [MIT License](LICENSE).
