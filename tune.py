from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.autograd as autograd

import random
import chocolate as choco

from runexp import MyTrainable

def calculate_space_size(params, iteration_time = 5, verbose = 1):
	"""
	args:
		iteration_time: time in minutes
	"""
    space_size = 1
    for key in params["tunable_params"].keys():
        space_size *= len(params["tunable_params"][key])

	if verbose:
	    print("space_size:", space_size)
	    print("time_upper_bound:", space_size * iteration_time / 60 / 6, "hours")
    return space_size

if __name__ == "__main__":


	params = {
		# Fixed
		"num_trials": 5,
		"num_frames": 30000,
		"num_val_trials" : 10,
		"batch_size" : 32,
		"gamma" : 0.99,
		"cnn": False,
		"num_val_trials" : 10,
		"batch_size" : 32,
		"gamma" : 0.99,
		"num_trials" : 5,
		"USE_CUDA" : True,
		"device" : "",
		"eps": 1.,
		"num_workers": 12,
		"num_gpus": 1,
		# Searched over
		"hyperparam_mutations":{
			"method": choco.choice(['our', 'PER']),
			"return_latent":choco.choice['state', 'first_hidden', 'second_hidden', 'last']),
			"var": choco.choice([.3, 1., 3., 10.]),
			"mean": choco.choice([0.]),
			"decision_eps": choco.choice([1.]),
			"theta" : choco.choice([1.]),
			"invert_actions" : choco.choice([False]),
			"lr": choco.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
			},
		}

    # initialize chocolate
	i = 0
	sampler = None
	while sampler is None:
	    try:
	        # connect
	        database_dir = "sqlite:///roper_experiments" + "_".join(params["tunable_params"].keys()) + str(i) + ".db"
	        conn = choco.SQLiteConnection(database_dir)
	        sampler = choco.QuasiRandom(conn, params["hyperparam_mutations"], seed=2)
	        print("saving results in", database_dir)
	    except:
	        i += 1
	        pass

	k = 1
	for i in range(calculate_space_size(params)):
	    # Get one hyperparameter configuration from the space
	    token, next_params = sampler.next()
	    # Combine Tunable and Fixed Hyperparameters.
	    hyperparams = {**params, **next_params}
	    print("params", hyperparams)
	    # Create autoencoder model.
	    model = Trainable(hyperparams,token['_chocolate_id'], ...) # TODO: complete parameters
	    # Calculate Loss
        loss, val_loss = model.run_config()
        print(" Loss:",loss, "\n val_loss", val_loss)
	    # Update Database
	    sampler.update(token, val_loss)
