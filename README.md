# uncertainty-esm

This repository is a playground to test minimal-example implementations of uncertainty methods for the energy system model **PyPSA**. Currently explored are different packages with monte-carlo approaches that are useful for **global sensitivity analysis**.

- scripts/test.py, is the script which implements different monte-carlo approaches
- config.yaml, is the input sheet

The repository implements Monte-Carlo approaches from the following packages:
- chaospy
- pyDOE
- scipy

Each of the above packages has much more to offer. Feel free to contribute or point-towards alternative methods.

Installation
------------

```bash
conda create --name uncertainty-esm -y
conda activate uncertainty-esm
pip install pypsa pyDOE chaospy pyyaml
```

To test the current default implementation run:
```bash
python scripts/test.py
```
This will create 169 optimizations `result0.nc`, `result1.nc`,... `result168.nc` (takes about 10min)
Make adjustment in the config.yaml to explore other methods/input options.
