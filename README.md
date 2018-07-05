Code to simulate a randomly connected network of excitatory (simplified) HH neurons, with STSP, and metabolic regulation.

The core of the code is an adaptation of the [MATLAB code from the Guerrier et al. (2015) PNAS paper](http://bionewmetrics.org/category/projects/neuroscience/).

There are two main differences, however:

1) Here we only simulate a single pool of vesicles (the docked ones)
2) A novel metabolic function modulating the docking rate is included.
