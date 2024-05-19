# Triplane config
decoder = dict(
    whiteout = True, 
    white_background = True,
    density_activation = "trunc_exp",
    hidden_dim = 128,
    hidden_layers = 5,
    hn = 0,
    hf = 60,
    nb_bins = 64,
    nb_bins_sample = 64,
    train_stratified = True,
    testing_batch_size = 2048*4,
)
