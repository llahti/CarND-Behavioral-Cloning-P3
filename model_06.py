"""
This model is based on model_05 and purpose is to 
try to minimize the size of model.
"""

from keras import optimizers
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.models import Sequential


def create_model(input_shape=(299, 299, 3)):
    """
    This model expects input images to be shape (75, 300, 3)

    :return: 
    """

    model = Sequential()

    # Constants
    model.name = 'model_06'
    input_shape = (75, 300, 3)

    # Mean_pool image to 75,75
    model.add(AveragePooling2D(pool_size=(1, 4), input_shape=input_shape, name='01_MaxPool'))

    model.add(Conv2D(4, 3, 3, subsample=(2,2), activation='elu', name='02_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(4, 3, 3, subsample=(1,1), activation='elu', name='03_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(6, 3, 3, subsample=(1,1), activation='elu', name='04_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(6, 3, 3, subsample=(1,1), activation='elu', name='05_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(8, 3, 3, subsample=(1,1), activation='elu', name='06_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(8, 3, 3, subsample=(1, 1), activation='elu', name='07_Conv_'))
    model.add(Dropout(0.1))

    model.add(Conv2D(8, 3, 3, subsample=(1, 1), activation='elu', name='08_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(10, 3, 3, subsample=(1, 1), activation='elu', name='09_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(16, 3, 3, subsample=(1, 1), activation='elu', name='10_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(24, 3, 3, subsample=(1, 1), activation='elu', name='11_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(24, 3, 3, subsample=(2, 1), activation='elu', name='12_Conv'))
    model.add(Dropout(0.1))

    model.add(Conv2D(24, 3, 3, subsample=(2, 1), activation='elu', name='13_Conv'))
    model.add(Dropout(0.1))

    # Flatten 3D-tensors to 1D
    model.add(Flatten())
    model.add(Dense(1440, activation='elu', name="14_FC1"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='elu', name="15_FC2"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='elu', name="16_FC3"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu', name="17_FC4"))
    model.add(Dropout(0.1))
    # Use sigmoid for cleaner activations
    model.add(Dense(10, activation='sigmoid', name="18_FC5"))
    # The regression output layer, (Steering Angle)
    model.add(Dense(1))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
    model.compile(optimizer=adam, loss='mean_squared_error' )

    return model

if __name__ == "__main__":
    create_model().summary()
