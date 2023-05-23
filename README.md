This is a repository with  experiments for the paper **Error Feedback Shines when Features are Rare**

This repository includes source code and guidelines for reproducing experiments for this paper.

## Prerequisites

The experiments have been constructed via modifying FL_PyTorch [https://arxiv.org/abs/2202.03099](https://arxiv.org/abs/2202.03099)

This simulator is constructed based on the PyTorch computation framework. The first step is preparing the environment. 

If you have installed [conda](https://docs.conda.io/en/latest/) environment and package manager then you should perform only the following steps for preparing the environment.

```
conda create -n fl python=3.9.1 -y
conda install -n fl pytorch"=1.10.0" torchvision numpy cudatoolkit"=11.1" h5py"=3.6.0" coloredlogs matplotlib psutil pyqt pytest pdoc3 wandb -c pytorch -c nvidia -c conda-forge -y
conda activate fl
```

Our experiments have been carried out utilizing computation in CPUs.

Our modification of the simulator is located in `./fl_pytorch`. Use this version that we're providing instead of the Open Source version.

Also, your OS should have installed a BASH interpreter or its equivalent.

## Place with Execution Command Lines

Change the working directory to `"./fl_pytorch"`. The sequence of scripts to execute:


* [compute_start_gradient_at_zero_non_cvx_experiments.sh](fl_pytorch/compute_start_gradient_at_zero_non_cvx_experiments.sh)


This script computes gradients at first iterates for non-convex experiments and use this information during plotting.


* [compute_start_gradient_at_zero_for_real_ds.sh](fl_pytorch/compute_start_gradient_at_zero_for_real_ds.sh)

This script computes gradients at first iterate for convex experiments and uses this information during plotting for the gradient at the start.


* [compute_start_gradient_at_zero_cvx_experiments.sh](fl_pytorch/compute_start_gradient_at_zero_cvx_experiments.sh)

This script computes gradients at first iterate for non-convex experiments and uses this information during plotting.

* [plot_c_patterns.sh](fl_pytorch/plot_c_patterns.sh)

Script to generate sparsity patterns interactively


* [scripts_for_libsvm_datasets](fl_pytorch/scripts_for_libsvm_datasets)

Scripts for experiments with LIBSVM Datasets.

* [scripts_for_synthetics_noncvx](fl_pytorch/scripts_for_synthetics_noncvx)

Scripts for synthesized experiments for Non-Convex Optimization.

* [scripts_for_synthetics_cvx](fl_pytorch/scripts_for_synthetics_cvx) 

Scripts for synthesized experiments for Convex Optimization (special controllable linear regression)

## Tracking results online

If you want to use [WandB](https://wandb.ai/settings) online tool to track the progress of the numerical experiments please specify:
* `--wandb-key "xxxxxxxxxxx" ` with a key from your wandb profile: [https://wandb.ai/settings](https://wandb.ai/settings
* `--wandb-project-name "vvvvvvvvvv"` with a project name that you're planning to use.
You should replace `--wandb-project-name "vvvvvvvvvv"` with a project name that you're planning to use or leave the default name. Both of these keys should be replaced manually if you're interested in WandB support.

## Visualization of the Results

The result in binary files can be loaded into the simulator `fl_pytorch\fl_pytorch\gui\start.py`. After this plots can be visualized in *Analysis* tab. 

Recommendations on how to achieve this are available in TUTORIAL for this simulator.

To automatically generate names please add `{comment}` into Label generation. For the purpose of publication plots in the paper, we have used line size 4, and font sizes 37.
