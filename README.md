# Flexible unsupervised binary change detection algorithm identifies phase transitions in continuous image streams

### Paul Chao <sup>1</sup> , Xianghui Xiao <sup>2</sup>, and Ashwin J. Shahani <sup>1</sup> 

<sup>1</sup> University of Michigan, Ann Arbor
<sup>2</sup> Brookhaven National Laboratory, Upton, NY

**Abstract**: Sequences of projection images collected during in-situ tomography experiments can capture the formation of
patterns in crystallization and yield their three-dimensional growth morphologies. These image streams generate
enormous and high dimensional datasets that span the full extent of a phase transition. Detecting from the continuous
image stream the characteristic times and temperatures at which the phase transition initiates is a challenge because
the phase change is often swift and subtle. Here we show a flexible unsupervised binary classification algorithm
to identify a change point during data intensive experiments. The algorithm makes a prediction based on statistical
metrics and has a quantifiable error bound. Applied to two in-situ X-ray tomography experimental datasets collected
at a synchrotron light source, the developed method can detect the moment at which the solid phase emerges from the
parent liquid phase upon crystallization and without performing computationally expensive volume reconstructions.
Our approach is verified using a simulated X-ray phantom and its performance evaluated with respect to solidification
parameters. The method presented here can be broadly applied to other big data problems where time series can be
classified without the need for additional training data.  

Keywords: Solidification, nucleation, machine learning, sinogram, x-ray tomography

## Installation

Install the enviroment dependencies by running:

```python
$ conda env create -f environment.yaml
```

## Usage

Explore the Example.ipynb jupyter notebook to get started. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
