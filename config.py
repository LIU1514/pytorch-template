args = dict()


#project structure
args["CODE_DIRECTORY"] = "absolute path to code directory"
args["DATA_DIRECTORY"] = "absolute path to data directory"
args["PRETRAINED_MODEL_FILE"] = "relative path to pretrained model file"
args["TRAINED_MODEL_FILE"] = "relative path to trained model file"


#data
args["VALIDATION_SPLIT"] = 0.05
args["NUM_WORKERS"] = 8


#training
args["SEED"] = 10
args["BATCH_SIZE"] = 8
args["NUM_EPOCHS"] = 15
args["SAVE_FREQUENCY"] = 5


#optimizer and scheduler
args["LEARNING_RATE"] = 0.001
args["MOMENTUM1"] = 0.9     
args["MOMENTUM2"] = 0.999 
args["LR_DECAY"] = 0.95


#loss function
args["LAMBDA"] = 10



if __name__ == '__main__':
    
    for key,value in args.items():
        print(str(key) + " : " + str(value))