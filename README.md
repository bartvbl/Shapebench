# ShapeBench: a new approach to benchmarking local 3D shape descriptors

This is the code repository for the evaluation strategy described in the paper:

van Blokland BI. ShapeBench: A new approach to benchmarking local 3D shape descriptors. Computers & Graphics. 2024 Aug 22:104052.

[Link to paper](https://doi.org/10.1016/j.cag.2024.104052)

### Getting started

Clone the repository using the --recursive flag:
```
git clone https://github.com/bartvbl/ShapeBench --recursive
```
You should subsequently run the python script to install dependencies, compile the project, and download precompiled cache files. You can use the following command to do so:
```
python3 replicate.py
```
Refer to the included PDF file for information about replicating the results produced for the paper.

### System requirements
* 32GB of RAM (64GB is highly recommended)
* The project has been tested on Linux Mint 21.3.
* The CUDA SDK must be installed, but no GPU is required to run the benchmark

There's nothing that should prevent the project to be compiled on Windows, but it has not been tested.

While many of the experiments and filters will run on a system with 32GB of RAM, you will likely need to apply a number of thread limiters in order to reduce memory requirements. Make in any case sure to have enough swap space available, and a healthy dose of patience when using thread limiters.

### Abstract

The ShapeBench evaluation methodology is proposed as an extension to the popular Area Under Precision-Recall Curve (PRC/AUC) for measuring the matching performance of local 3D shape descriptors. It is observed that the PRC inadequately accounts for other similar surfaces in the same or different objects when determining whether a candidate match is a true positive. The novel Descriptor Distance Index (DDI) metric is introduced to address this limitation. In contrast to previous evaluation methodologies, which identify entire objects in a given scene, the DDI metric measures descriptor performance by analysing point-to-point distances. The ShapeBench methodology is also more scalable than previous approaches, by using procedural generation. The benchmark is used to evaluate both old and new descriptors. The results produced by the implementation of the benchmark are fully replicable, and are made publicly available.

### Citation

```
@article{van2024shapebench,
  title={ShapeBench: A new approach to benchmarking local 3D shape descriptors},
  author={van Blokland, Bart Iver},
  journal={Computers \& Graphics},
  pages={104052},
  year={2024},
  publisher={Elsevier}
}
```

### Troubleshooting
To ensure compatibility with the default LibEigen3 installation from `apt`, please modify line `27` of [CMakeLists](https://github.com/bartvbl/ShapeBench-Replication-Archive/blob/main/CMakeLists.txt) to
```
find_package(Eigen3 3.3.0 REQUIRED)
```
