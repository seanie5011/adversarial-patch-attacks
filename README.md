# adversarial-patch-attacks

`datasets.py` - classes for importing various datasets used, with some testing functionality

`transfer_model.py` - functionality for transfer learning using resnet18

`digital_attacks.py` - performing the digital patch attack using the transferred model and dataset

`digital_attacks_test.py` - testing the digital patch attack using the transferred model and dataset

`utils.py` - general utilities: testing and logging

`datasets/` - where the various datasets used are stored, see README for more

`logs/` - where the log files for the patch attacks are stored

`models/` - where the various models used are stored, see README for more

`training/` - where the output of training runs are stored, see README for more

Currently the only uploaded output is that in `training/run3/`, where the output of the system is shown attempting a targeted black-box digital attack to trick the resnet18 model to see bees in the hymenoptera dataset.