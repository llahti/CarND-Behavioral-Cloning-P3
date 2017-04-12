"""This file is used to train model."""
from generator import GenTrainTestSplit, BatchGenerator
from generator import GenGaussVerticalShift, GenRandHorizontalShift, GenFlipLR
from generator import GenRandBrightness, FileReaderCarSim, GenPreprocess
from utils import get_session, KTF
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import model_04
import model_05
import model_06
from utils import plot_keras_history

import numpy as np


def augment_pipeline(generator):
    """
    This is the data augmentation pipeline.
    
    :param generator: generator which will return tuple of (x, y) where 
        x is 4D-tensor. 
    :return: augmented data as a tuple (x, y) where x is image tensor and y steering angle
    """
    gen = generator
    # Add preprocessing here as it is computationally intensive,
    gen = GenPreprocess(gen)
    # All steps staring from here are for data augmentation
    gen = GenRandBrightness(gen, br_range=(0.2, 1.))
    gen = GenGaussVerticalShift(gen, mu=0, sigma=8, multiplication=2)
    gen = GenRandHorizontalShift(gen, (-20, 20), multiplication=2, target_distance=50)
    gen = GenFlipLR(gen)
    return gen


def train_model(model, batch_size = 64, nb_epoch=10, verbose=1):
    """
    This methods handles model training.
    
    :param model: keras model object 
    :param batch_size: training batch size
    :param nb_epoch: number of epochs
    :param verbose: see keras documentation.. 0=quit, 1 verbose, 2 show only epoch results
    :return: trained model
    """
    # Establish CSV file reader and basic Generators
    train_valid_reader = FileReaderCarSim(driving_log_file)
    # Split into training and validation sets
    train_gen, test_gen = GenTrainTestSplit(train_valid_reader,
                                            test_size=0.05, rollover=True)
    # Augment training set data
    train_gen = augment_pipeline(train_gen)
    # Preprocess training and test
    # train_gen = GenPreprocess(train_gen)
    test_gen = GenPreprocess(test_gen)
    # Create batches
    train_gen = BatchGenerator(train_gen, batch_size=batch_size)
    test_gen = BatchGenerator(test_gen, batch_size=batch_size)

    # checkpointer handles saving models when val_loss have been improved
    filename = "./checkpoints/" + model.name + "_e{epoch:03d}_vl{val_loss:.3f}.h5"
    checkpointer = ModelCheckpoint(filepath=filename, verbose=1,
                                   monitor='val_loss', save_best_only=False)


    # Reserve about 1/3 of GPU memory
    sess = get_session(gpu_fraction=0.3)
    KTF.set_session(sess)

    hist = model.fit_generator(train_gen, samples_per_epoch=len(train_gen), nb_epoch=nb_epoch,
                               verbose=verbose,
                               validation_data=test_gen, nb_val_samples=len(test_gen),
                               max_q_size=batch_size*2, nb_worker=3, pickle_safe=True,
                               callbacks=[checkpointer],
                               initial_epoch=0)

    return model, hist


if __name__ == "__main__":
    #TODO: Modify this so that it is possible to supply command line parameters
    # 1. to select model
    # 2. to decide whether existing model training will be continued

    training_data_path = "./training_data/train3"
    driving_log_file = training_data_path + "/driving_log.csv"
    #driving_log_file = training_data_path + "/driving_log_test.csv"

    # Define filenames to save model and weights
    model_name = "model_06"
    model_filename =  model_name + ".h5"
    model_weight_filename = model_name + "_weights.h5"

    # Create or load the model
    #model = model_04.create_model(input_shape=(75, 300, 3))
    model = model_06.create_model()
    #model.summary()
    #model = load_model(model_filename)
    #model.load_weights(model_weight_filename)
    model.name = model_name

    # Training operation
    model, hist = train_model(model, batch_size=32, nb_epoch=50)
    # Plot training and validation losses
    plot_keras_history(hist)

    # Save model
    print("Saving model as ", model_filename)
    model.save(model_filename)
    print("Saving model weights into file, ", model_weight_filename)
    model.save_weights(model_weight_filename)

