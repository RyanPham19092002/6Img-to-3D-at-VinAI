optimizer = dict(
    lr=5e-3,   #orgin 5e-5
    num_training_steps=1000,
    num_epochs = 200,
    lpips_loss_weight = 0.2,
    tv_loss_weight = 0.,
    dist_loss_weight = 1.0e-03,
    clip_grad_norm=1.5,  #orgin 1.5
    depth_loss_weight=1
    # depth_loss_weight=1
)