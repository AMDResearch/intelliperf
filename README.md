# Maestro

Orchestrating the Omniverse.



## Getting Started

We provide an Apptainer image containing all the dependencies. To get started, run:
```
./apptainer/build.sh
```

To start the Apptainer container, run:

```
./apptainer/run.sh
```

### Installation

First, install the many dependencies:
```terminal
python3 scripts/install_tool.py --all --clean
```

Then, install Maestro:

```
pip install -e .
```
## Demos

To run a simple demo:

```
cd src
python3 demo.py
```