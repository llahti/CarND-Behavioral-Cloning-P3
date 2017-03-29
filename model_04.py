from keras import optimizers
from keras.layers import Cropping2D, Conv2D, Dense, Dropout, Flatten, Lambda
from keras.layers import AveragePooling2D
from utils import k_utils
from keras.models import Sequential



def create_model(input_shape=(299, 299, 3)):
    """
    This model is modified so that it expects preprocessed images.

    :return: 
    """

    model = Sequential()
    model.name = 'model_04'

    # Mean_pool image to 75,75
    model.add(AveragePooling2D(pool_size=(1, 4), input_shape=input_shape))

    # Color information selection by neural net
    model.add(Conv2D(4, 1, 1, subsample=(1,1), activation='sigmoid', name='Conv_01'))

    # Normalize
    #model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-5))

    model.add(Conv2D(4, 3, 3, subsample=(2,2), activation='elu', name='Conv_02'))
    model.add(Dropout(0.1))

    model.add(Conv2D(6, 3, 3, subsample=(1,1), activation='elu', name='Conv_03'))
    model.add(Dropout(0.1))

    model.add(Conv2D(9, 3, 3, subsample=(1,1), activation='elu', name='Conv_04'))
    model.add(Dropout(0.1))

    model.add(Conv2D(9, 3, 3, subsample=(1,1), activation='elu', name='Conv_05'))
    model.add(Dropout(0.1))

    # output 1, 16, 64
    model.add(Conv2D(12, 3, 3, subsample=(1,1), activation='elu', name='Conv_06'))
    model.add(Dropout(0.1))

    model.add(Conv2D(12, 3, 3, subsample=(1, 1), activation='elu', name='Conv_07'))
    model.add(Dropout(0.1))

    model.add(Conv2D(16, 3, 3, subsample=(1, 1), activation='elu', name='Conv_08'))
    model.add(Dropout(0.1))

    model.add(Conv2D(24, 3, 3, subsample=(1, 1), activation='elu', name='Conv_09'))
    model.add(Dropout(0.1))

    model.add(Conv2D(24, 3, 3, subsample=(1, 1), activation='elu', name='Conv_10'))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, 3, 3, subsample=(1, 1), activation='elu', name='Conv_11'))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, 3, 3, subsample=(2, 1), activation='elu', name='Conv_12'))
    model.add(Dropout(0.1))

    model.add(Conv2D(48, 3, 3, subsample=(2, 1), activation='elu', name='Conv_13'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(2880, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    # Use sigmoid for cleaner activations
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dropout(0.1))
    # Use sigmoid for cleaner activations
    model.add(Dense(10, activation='sigmoid'))
    # The regression output layer, (Steering Angle)
    model.add(Dense(1))


    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # Note! Change to this below optimized config had huge effect on training.
    # Now it is steadily converging, but still can't drive
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
    model.compile(optimizer=adam, loss='mean_squared_error' )

    return model
