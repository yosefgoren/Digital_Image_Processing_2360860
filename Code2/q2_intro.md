# Question 2
For our solution we created a simple unet implemenation (see `unet.py`) and used the small SRGB variant of the sidd dataset (see `sidd.py`).

All of the source files relating to this question are provided.
The core of the training code is at `train.py`.
The hyperparameters were specified by the `training_specification.json` file (not attached). Here is it's content:
```json
{
    "patch_size": 128,
    "patches_per_image": 16,
    "batch_size": 32,
    "epochs": 16,
    "learning_rate": 0.0001,
    "optimizer": "Adam",
    "loss_function": "L1Loss",
    "model_type": "UNet",
    "device": "xpu",
    "dataset_path": "/home/yogo/media/Datasets/SIDD_Small_sRGB_Only/Data",
    "max_image_pairs": 60,
    "checkpoint_frequency": 1
}
```

You can find all of the training statistics in the `metrics.json` file.
We have also attached a file named `unet_sidd.pth` with the weights of the trained model.

Below is an overview of our question 2 results, given in two parts:

1. Model training (the successful training run that we chose):
    * A graph showing the loss over time.
    * A list of images that visually show the progress of the denoising model over the training process (each epoch). `Denoised` is our model's output.
2. Model testing - an overview of the performance of the model we have trained.
    * The numerical results of testing the model.
    * A few sample images that visually showcase the model's performance (`Denoised` is our model's output).