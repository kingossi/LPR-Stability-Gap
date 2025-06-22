# LPR-Stability-Gap

> **Note:** This project is heavily inspired by Gido van de Ven’s code available at [https://github.com/GMvandeVen/continual-learning/tree/master/StabilityGap](https://github.com/GMvandeVen/continual-learning/tree/master/StabilityGap).  
> It implements **Layerwise Proximal Replay (LPR)** as proposed by the PLAI group ([https://github.com/plai-group/LPR](https://github.com/plai-group/LPR)).

This repository provides an implementation of **LPR** for continual learning, applied to the Rotated MNIST benchmark. The approach evaluates the stability of learning under task switches by combining replay-based training with layerwise activation-based preconditioning.

## Features

- Continual learning setup using rotated versions of MNIST (0°, 60°, 120°, 180°)
- MLP architecture with optional layerwise activation tracking
- Replay buffer to mitigate forgetting between tasks
- Computation of LPR preconditioners based on stored activations
- Evaluation of test accuracy over time, visualized across tasks
- Multi-run experiment grid over various learning rates and LPR weights
- Results are saved to disk and visualized in a PDF (`lpr_results_plots.pdf`)

## Usage

Simply run the script with Python 3. Training results are saved incrementally using `pickle`, and can be recovered if interrupted.

## Requirements

To install the dependencies, run:

```bash
pip install -r requirements.txt
