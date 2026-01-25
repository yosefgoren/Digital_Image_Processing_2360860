## Programs
* train.py: Trains the model (or resumes training)
* generate_report.py: Generate a PDF that shows the training process for a given run (an execution of run.py)
* test.py: Load the trained model for a given run, apply it to some pictures and generate a PDF that showcases the results.


## Dataset
Was downloaded from https://abdokamel.github.io/sidd/
The variant used is the `sRGB` one - composed of PNG's.

## Training Configuration
The training program assumes there will be a configuration file named `training_specification.json`.
Here is an example of it's content correct to the time of this writing:
```json
{
  "patch_size": 128,
  "patches_per_image": 16,
  "batch_size": 32,
  "epochs": 16,
  "learning_rate": 0.0001,
  "optimizer": "Adam",
  "loss_function": "MSELoss",
  "model_type": "UNet",
  "device": "xpu",
  "dataset_path": "/home/yogo/media/Datasets/SIDD_Small_sRGB_Only/Data",
  "max_image_pairs": 100,
  "checkpoint_frequency": 1
}
```

This format is specified by the `pydantic.BaseModel` class named `TrainingRun` found at `metrics.py`.