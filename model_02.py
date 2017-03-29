from keras import optimizers
from keras.layers import Cropping2D, Conv2D, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential


def image_preprocess(x):
    """
    This is a image preprocessing layer for keras.
    it expects that data value range is 0...255 because there is hardcoded division by 127

    1. Range
    """
    # x = K.cast(x, dtype='float32')
    x = x / 127. - 1
    return x

def create_model_02(input_shape=(299, 299, 3)):
    """
    https://keras.io/applications/

    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

    :return: 
    """

    model = Sequential()
    model.name = 'model_02'

    # Remove hood and sky by using
    # Crop image ((top_crop, bottom_crop), (left_crop, right_crop))
    # output size (100, 320, 3)
    model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                         input_shape=input_shape))

    # Image pre-processing
    model.add(Lambda(image_preprocess, name='preprocess'))

    # Color information selection by neural net
    model.add(Conv2D(3, 1, 1, subsample=(1,1), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(48, 3, 3, subsample=(1,2), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, 3, 3, subsample=(1,1), activation='elu'))
    model.add(Dropout(0.1))

    # output 1, 16, 64
    model.add(Conv2D(128, 3, 3, subsample=(1,1), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, 3, 3, subsample=(1, 1), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, 3, 3, subsample=(2, 2), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, 3, 3, subsample=(1, 1), activation='elu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    # 20170405 change 5376  --> 4992
    model.add(Dense(4992, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='linear'))
    # The regression output layer, (Steering Angle)
    model.add(Dense(1))
    # By using internal presentation of radians we can limit
    # values to range -0.436...0.436. scale by 2 to get range ~ -1...1
    # Finally convert radians to degrees.
    model.add(Lambda(lambda x: x * 57.295779))
    #model.add(Lambda(lambda x: 2 * x)))

    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # Note! Change to this below optimized config had huge effect on training.
    # Now it is steadily converging, but still can't drive
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
    model.compile(optimizer=adam, loss='mean_squared_error' )

    return model