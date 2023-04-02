# Leveraging Graph Networks to Model Environments in Reinforcement Learning
This code accompanies the FLAIRS '23 paper "Leveraging Graph Networks to Model Environments in Reinforcement Learning", by Viswanath Chadalapaka, Volkan Ustun, and Lixing Liu.

## To run all of our experiments
1. Install anaconda and the relevant dependencies via env.yml. If env.yml doesn't work, then installing the dependencies from env-simple.yml should also be sufficient to run the project (although without the exact libraries we used, you may get slightly different results from our work).
2. Run `python run_experiments.py`.

## To run specific experiments
Examine the structure of the `configs/experiments.json` file. Each of the json blocks in that file represents an experiment that `run_experiments.py` will run, where the key/value pairs that can be added correspond directly to the flags available for `train.py` (run `python train.py --help` to see all of these flags).

## Metrics
There are two options for seeing metrics. The first is to run the collect_stats.ipynb notebook -- if you are on linux, you can simply run through the collect_stats.ipynb notebook after running the experiments and you should see outputs. Otherwise, you may have to edit the notebook to point to the ray_results directory for your OS (on linux, this is `~/ray_rllib`).

The second option is to view the results directly in tensorboard. You can run tensorboard to see metrics using `tensorboard serve --logdir logs_directory`, where `logs_directory` is the location of the ray_results folder for your OS (again, this should be something like `~/ray_rllib`).

## More tests
To see specific model behavior and specific locations, try running `test.py`. test.py reads the model checkpoints that are stored in the checkpoints/ folder during training, and runs those for specific starting locations in the simulation.
