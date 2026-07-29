# Gradient Pattern Analysis (GPA) - Python Implementation

This repository contains a Python adaptation of the **Gradient Pattern Analysis (GPA)** method for characterizing spatial patterns through gradient fields, phase information, vector gradient moments, and Delaunay triangulation.

The implementation provides tools for analyzing different spatial structures, including symmetric, asymmetric, and laminar patterns, as well as computing GPA-related descriptors.

---

# Scientific Background and Implementation Notes

This Python implementation of **Gradient Pattern Analysis (GPA)** is an adaptation of the original Cython implementation developed in the **CyMorph** project:

Repository:
https://github.com/rsautter/CyMorph

The objective of this project is to provide a Python-based implementation of the GPA methodology while preserving the mathematical formulation and computational procedures defined in the original implementation.

The equations, definitions, and GPA descriptors implemented here follow the methodology presented in:

- Rosa et al. (2018) — *Gradient pattern analysis applied to galaxy morphology*
- Barchi et al. (2020) — *Machine and Deep Learning applied to galaxy morphology - A comparative study*
- Kolesnikov et al. (2024) — *Unveiling galaxy morphology through an unsupervised-supervised hybrid approach*

The implementation follows the main GPA computational steps:

- computation of spatial gradients;
- calculation of gradient magnitude and phase;
- selection of significant gradient structures using tolerance parameters;
- construction of gradient vector patterns;
- Delaunay triangulation of gradient structures;
- computation of GPA descriptors and gradient moments.

This implementation is intended for research and educational purposes, enabling integration with modern Python scientific libraries and astronomical image-processing workflows.

---

# Author

**Carlos Eduardo Falandes**

MSc Student in Applied Computing  
National Institute for Space Research (INPE)

---

# Requirements

The project was developed using **Python 3.11** with the following scientific libraries:

- NumPy
- SciPy
- Matplotlib

Additional libraries may be required depending on the application pipeline.

---

# Installation

## 1. Create a Conda environment

Create a new environment:

```bash
conda create -n gpa_env python=3.11
```

Activate the environment:

```bash
conda activate gpa_env
```

---

## 2. Install dependencies

Install the required packages:

```bash
conda install -c conda-forge numpy scipy matplotlib
```

For astronomical image analysis, the following packages are recommended:

```bash
conda install -c conda-forge astropy sep scikit-image pandas jupyter
```

---

# Usage

## Import the required modules

```python
import numpy as np
import matplotlib.pyplot as plt

from gpa import GPA
```

---

# Creating Input Matrices

The GPA method can be applied to different spatial patterns.

## Symmetric Matrix

Example:

```python
symmetric = create_symmetric_matrix(5)

print(symmetric)
```

Output:

```
[[1 1 1 1 1]
 [1 2 2 2 1]
 [1 2 3 2 1]
 [1 2 2 2 1]
 [1 1 1 1 1]]
```

This matrix represents a radially symmetric structure.

---

## Asymmetric Matrix

Example:

```python
asymmetric = create_asymmetric_matrix(5, seed=42)

plt.imshow(asymmetric)
plt.show()
```

This generates a matrix with randomly distributed values, producing an asymmetric spatial pattern.

---

## Laminar Matrix

Example:

```python
laminar = create_laminar_matrix(5)

plt.imshow(laminar)
plt.show()
```

This creates a structured pattern with increasing values along rows.

---

# Running GPA Analysis

Create a GPA object:

```python
gpa_object = GPA(matrix)
```

where:

- `matrix` is the input two-dimensional array.

Example:

```python
gpa_object = GPA(symmetric)
```

---

# Setting the Pattern Center

The reference position can be defined using:

```python
gpa_object.setPosition(x, y)
```

Example:

```python
gpa_object.setPosition(2, 2)
```

For astronomical images, the center can be automatically defined as the brightest pixel:

```python
max_position = np.unravel_index(
    np.argmax(image),
    image.shape
)

gpa_object.setPosition(
    max_position[1],
    max_position[0]
)
```

---

# Evaluating GPA

The GPA analysis is performed using:

```python
results = gpa_object.evaluate(
    mtol=0.02,
    ftol=0.03,
    ptol=0.01
)
```

Parameters:

| Parameter | Description |
|-----------|-------------|
| `mtol` | Gradient magnitude tolerance |
| `ftol` | Gradient phase tolerance |
| `ptol` | Position tolerance |

---

# Example Workflow

```python
import numpy as np
from gpa import GPA

matrix = create_symmetric_matrix(100)

gpa = GPA(matrix)

gpa.setPosition(
    matrix.shape[1] // 2,
    matrix.shape[0] // 2
)

results = gpa.evaluate(
    mtol=0.02,
    ftol=0.03,
    ptol=0.01
)

print(results)
```

---

# Visualization

The input pattern can be visualized using:

```python
plt.figure(figsize=(5,5))

plt.imshow(
    matrix,
    cmap="viridis",
    origin="upper"
)

plt.colorbar(label="Value")
plt.title("Input pattern")

plt.show()
```

---

# Saving the Environment

To save the current environment:

```bash
conda env export > environment.yml
```

The environment can be recreated using:

```bash
conda env create -f environment.yml
```

---

# Acknowledgements

This work is based on the original GPA implementation developed in the **CyMorph** project:

https://github.com/rsautter/CyMorph

The author acknowledges the developers of CyMorph and the authors of the scientific publications that established and extended the Gradient Pattern Analysis methodology.

---

# References

## Rosa et al. (2018)

Rosa, R. R., de Carvalho, R. R., Sautter, R. A., et al.

**Gradient pattern analysis applied to galaxy morphology**

*Monthly Notices of the Royal Astronomical Society: Letters*,  
477(1), L101–L105 (2018).

DOI:

https://doi.org/10.1093/mnrasl/sly054

---

## Barchi et al. (2020)

Barchi, P. H., de Carvalho, R. R., Rosa, R. R., et al.

**Machine and Deep Learning applied to galaxy morphology - A comparative study**

*Astronomy and Computing*,  
30, 100334 (2020).

DOI:

https://doi.org/10.1016/j.ascom.2019.100334

---

## Kolesnikov et al. (2024)

Kolesnikov, I., Sampaio, V. M., de Carvalho, R. R., et al.

**Unveiling galaxy morphology through an unsupervised-supervised hybrid approach**

*Monthly Notices of the Royal Astronomical Society*,  
528(1), 82–107 (2024).

DOI:

https://doi.org/10.1093/mnras/stad3934