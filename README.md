# Implementation for learning curve extrapolation using graph-aware ordinary differential equations
The code is adapted from [https://github.com/google-research/torchsde](https://github.com/google-research/torchsde).
- To train and test the neural ODE model for learning curve extrapolation, run `python latent_de.py` with desired configuration. An example command is
  ```bash
  python3 latent_de.py --latent_dim 16 --cuda_id 3 --data cnn --batch-size 3 --diff_eq ode --seed $i --source_data cifar10 --pause-iters 50 --train_test_ratio 0.75
  ```
- To train and test LC-GODE for learning curve extrapolation, run `python latent_de_graph.py` with desired configuration.
  ```bash
  python3 latent_de_graph.py --latent_dim 16  --cuda_id 7 --data cnn --batch-size 3 --diff_eq ode --seed $i --source_data cifar10 --pause-iters 50 --train_test_ratio 0.75
  ```
The final evaluation metrics (e.g., mape, rmse, training runtime, test runtime) and the trained model will be saved in `args.train_dir`.
