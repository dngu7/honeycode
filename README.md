# HoneyCode

This repository is the official Pytorch implementation of HoneyCode, a machine learning system that uses a Tree Recurrent Network (TRN) and recurrent neural networks (RNNs) to generate synthetic software repositories for cybersecurity deception. 

This project implements the paper "HoneyCode: Automating Deceptive Software Repositories with Deep Generative Models" found @ TBA. 

## Installation

python3 + pytorch (>= 1.4.0)

The remainder of the packages can be installed with:

```pip install -r requirements.txt```

## Overview

In this repository, there are two folders: "train" and "run" which are described in the sections below. 
We recommend using a gpu to run the training and generation process. 

### Train Folder

The train folder holds the algorithms to train the 3 neural networks inside the overall repogen system. 

The models train on graphical representations of software repostories.
A usable example can be downloaded from this [link here](https://repogen.s3-ap-southeast-2.amazonaws.com/julia_graph_data.zip), or you may introduce your own in the same format. 
Make sure to place these inside the "data" folder in all training repositories. 
The example below refers to the path to place training data for treegen. 
```
  /repogen/train/treegen/data/julia
```

To demonstrate how to begin training, enter one of the folders (treegen) as an example.

Use main.py to kick off training with a gpu using the following command. 

```python main.py```


There are more custom options, such as selecting log levels, which can be found with the help flag. 

```python main.py --help```

Advanced settings can be modified in yaml files found under the 'config' folder. 

#### Evaluation / Generation of a single network

Once you have trained a network, setup a path to the new checkpoints in config as shown in the example below. 

```
  test_model_dir: ./exp/contentgen/GRUCharModel_Julia_2020-May-28-19-50-55_9760
  test_model_name: model_snapshot_0000004.pth
```

You can run evaluation of tests (treegen) or generation of samples (namegen/contentgen) using the following command.

```python main.py --device gpu --mode eval```

In namegen and contentgen, you have the option of creating different samples through two possible variables: seed and temperature. 
Decreasing temperature reduces the probability of randomness, and vice versa. 
Seeds can be altered through the config file. 

You may need to play around with the starting character to find the best match for each file extension. 

#### Pre-trained weights
Alternatively, if you are after a quick demo and want to skip training, you can download the checkpoint snapshots for a small network in the table below. 
Place the contents of these downloads into the 'run' directory under their respective 'models' folder. 


| Generator         | Download Link             | Save Location    |
| :---------------- |:-------------|:-----|
| Tree      | [Link](https://repogen.s3-ap-southeast-2.amazonaws.com/treegen.zip) | /repogen/run/models/treegen/ |
| Name      | [Link](https://repogen.s3-ap-southeast-2.amazonaws.com/namegen.zip)      |   /repogen/run/models/namegen/ |
| Content    | [Link](https://repogen.s3-ap-southeast-2.amazonaws.com/contentgen.zip)      |    /repogen/run/models/contentgen/ |



### Run Folder
The 'Run' folder contains the entry-point to run all 3 networks sequentially to generate samples of synthetic software repostiory. 

Ensure you have the correct path to the 'model_file' and 'model_snapshot' for each network properly setup in the yaml file found under 'config' folder. 
We recommend that you place these files under their respective 'models' folder found in ```/repogen/run/models/```.

The model_file should contain the model class and can be copied from the training folders. We leave the existing versions as an example.

Begin generation by entering into the 'Run' folder and running the following command.

```python main.py -s 1 -o '~/Downloads/' ```

 where the ```-s``` flag denotes the number of samples and the ```-o``` flag denotes the output location of the samples.
If you omit the output flag, then Repogen will default save to its experiment folder. 

#### Generation Process
The generation of the repostiory structure is visualized in the following figure. 
![gen_step](https://github.com/dngu7/myfiles/blob/master/generation_steps.png?raw=true)


### Samples
Below you will find samples created by each network. 

#### Tree Samples
![](https://github.com/dngu7/myfiles/blob/master/arb_samples_graphs.png?raw=true)

Graph visualization of samples from our \arb generative model trained over 100 epochs

#### Name Samples
![](https://github.com/dngu7/myfiles/blob/master/namegen_samples.png?raw=true)

Samples of directory names from name generative model. The left-column represents folder depth and the top-row represents the depth's respective sample id. Samples of depth=0 represents the names of parent directory.

#### Content Samples
![](https://github.com/dngu7/myfiles/blob/master/contentgen_sample.png?raw=true)

Julia sample snippet from the content generation network. This code was taken from a sample of 10000 characters and uses temperature of 0.8. Lower temperature results in more likely characters appearing. The sample looks visually authentic, however does not compile due to 3 missing variables inside the for-loop block. There are also unintelligible English phrases in the comments such as "Code code is cruce"..

## Acknowledgment
Treegen code was adapted from https://github.com/lrjconan/GRAN.
This work has been supported by Cybersecurity CRC.

## Reference us
Please cite the referenced paper above if this work has helped you in anyway. 

## Contact the author
Please feel free to contact d.d.nguyen@unsw.edu.au if you have any questions. 
If you find errors, please submit a github issue. 






