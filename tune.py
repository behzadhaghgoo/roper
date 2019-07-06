from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import ray
from ray.tune import run, Trainable, sample_from
from ray.tune.schedulers import PopulationBasedTraining

from runexp import MyTrainable

if __name__ == "__main__":

	pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_val_reward",
        mode="max",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "method": ['our', 'PER'],
            "return_latent":['state', 'first_hidden', 'second_hidden', 'last'],
            "var": [.3, 1., 3., 10.],
            "mean": [0.],
            "decision_eps": [1.],
            "theta" : [1.],
            "cnn": [False],
            "invert_actions" : [False],
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        })

	ray.init()
	run(
        MyTrainable,
        name="roper_test_0",
        scheduler=pbt,
        **{
            "config": {
                "num_trials": 5,
                "num_frames": 30000,
                "num_val_trials" : 10,
                "batch_size" : 32, 
                "gamma" : 0.99,
                "method": 'our',
	            "var": 1.,
	            "mean": 0.,
	            "decision_eps": [1.],
            	"theta" : [1.],
	            "cnn": False,
	            "invert_actions" : False,
	            "num_val_trials" : 10,
	            "batch_size" : 32,
	            "gamma" : 0.99,
	            "num_trials" : 5,
	            "USE_CUDA" : True,
	            "device" : "",
	            "eps": 1.,
                "num_workers": 0,
                "num_gpus": 0,
                # These params are tuned from a fixed starting value.
                "lr": 1e-4
                # These params start off randomly drawn from a set.
            },
        })
