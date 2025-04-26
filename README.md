# bayes-crm

## Research Paper
https://arxiv.org/abs/2407.01483

### Abstract
We propose a general-purpose approximation to the Ferguson-Klass algorithm for generating samples from Lévy processes without Gaussian components. We show that the proposed method is more than 1000 times faster than the standard Ferguson-Klass algorithm without a significant loss of precision. This method can open an avenue for computationally efficient and scalable Bayesian nonparametric models which go beyond conjugacy assumptions, as demonstrated in the examples section.

## Notebook examples
**analysis** - shows examples presented in the paper as well as the general methodology

Kaggle notebook [link](https://www.kaggle.com/code/dawidbernaciak/notebook70185d5f00)

**speed_comparison** - executes speed benchmarking vs state of the art methods in the literature 

Kaggle notebook [link](https://www.kaggle.com/code/dawidbernaciak/notebook169fa6903e)

## NBViewer
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/dbernaciak/bayes-crm/tree/main/notebooks/)

## Cloning the repository
This project uses [uv](https://docs.astral.sh/uv/getting-started/) as the dependency manager. To get started type ```uv sync``` to create a virtual environment with all dependencies installed.

## Kaggle
In order to run the notebooks on Kaggle, you need to:
* sign in to Kaggle
* follow the notebook links above
* click "Edit" button
* on the right hand side panel, click enable internet access
* run the notebook

## Runtime notes
Some of the analysis pieces and speed benchmarks are very time consuming to run and their outputs are dependent on the used hardware.
