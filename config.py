"""
Author: Smeet Shah

Description: File for setting values for the configuration options.

- Project structure paths
- Hyperparameters
"""

args = dict()


# project structure
args["CODE_DIRECTORY"] = "absolute path to code directory"
args["DATA_DIRECTORY"] = "absolute path to dataset directory"
args["DEMO_DIRECTORY"] = "absolute path to directory containing demo samples"
args["PRETRAINED_WEIGHTS_FILE"] = "/saved/weights/pretrained_weights.pt"
args["TRAINED_WEIGHTS_FILE"] = "/saved/weights/trained_weights.pt"

# data
args["VALIDATION_SPLIT"] = 0.05
args["NUM_WORKERS"] = 4

# training
args["SEED"] = 10
args["BATCH_SIZE"] = 4
args["NUM_EPOCHS"] = 75
args["SAVE_FREQUENCY"] = 5

# optimizer and scheduler
args["LEARNING_RATE"] = 0.001
args["MOMENTUM1"] = 0.9
args["MOMENTUM2"] = 0.999
args["LR_DECAY"] = 0.95

# loss
args["LAMBDA"] = 0.03


if __name__ == "__main__":

    for key, value in args.items():
        print(str(key) + " : " + str(value))
