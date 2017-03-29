"""
This file contains all the CSV file reading utilities to read images pointed by CSV file.
This file also contains data augmentation generators.
"""
import cv2
import numpy as np
import pandas as pd
import random
from scipy import misc
from sklearn.model_selection import ShuffleSplit

import threading
from utils import ImagePreprocess


class FileReaderCarSim:
    """
    An example how this file reader should be used in a mode which 
    goes through images once

    >>> reader = model.FileReaderCarSim(driving_log)
    """

    def __init__(self, filename, pictures=('center', 'left', 'right')):
        """
        :param filename: Path to driving log CSV file.
        :param pictures: Use this list to define which camera images you want to include. 
                         Options are ('center', 'left', 'right') 
        """

        # CSV file as a Pandas DataFrame
        self.drv_log = self._read_csv(filename)

        # Number of rows in CSV file
        self.csv_rows = len(self.drv_log.index)

        # Correction for left and right cameras, It'll be added to left and subtracted from right
        # self.columns = ('center', 'left', 'right')
        self.columns = pictures

        # Corrections for center, left and right cameras
        # Note! These are normalized numbers in range -1...1
        self.sta_correction = (0, 0.1, -0.1)

        # Defines how many samples there are totally available
        # Original images: (center, left, right) = 3
        self.images_per_row = len(self.columns)

        # Row counter
        self.i = 0

    def __getitem__(self, item):
        """
        This method indexes images according to CSV file. Actual order is defined by the 
        __init__ method parameter 'pictures'. It will define on which order images are read on rows.
        """
        # Calculate row and column indices
        row = int(item / self.images_per_row)
        nb_col = item % self.images_per_row

        # get column name
        column_name = self.columns[nb_col]  # use this to access DataFrame column

        # Read image
        img = misc.imread(self.drv_log[column_name][row]).astype(dtype=np.uint8)

        # Read corresponding steering angle and apply steering angle correction
        y = self.drv_log.steer_angle[row] + self.sta_correction[nb_col]

        return img, y

    def __iter__(self):
        return self

    def __len__(self):
        """
        :return: Total number of samples 
        """
        return self.csv_rows * self.images_per_row

    def __next__(self):
        """
         Gives next image and steering angle pair from CSV file.
         Image is read from disk.

         Note: This method is not thread safe.

        :return: tuple of (Image, y) 
        """

        if self.i >= len(self):
            # If reached end then raise StopIteration to indicate we are done
            raise StopIteration
        img, y = self[self.i]
        self.i += 1
        return img, y

    def _read_csv(self, filename):
        """
        Reads driving log from CVS file into pandas DataFrame.

        :param filename: path of file to read.
        :return: Pandas DataFrame (img_center, img_left, img_right, steer, throttle, break, speed) 
        """
        # Data columns in csv file
        column_names = ["center", "left", "right", "steer_angle", "throttle", "break", "speed"]

        # Column data types
        data_types = {"center": str, "left": str, "right": str,
                      "steer_angle": np.float32, "throttle": np.float32,
                      "break": np.float32, "speed": np.float32}

        return pd.read_csv(filename, sep=',', names=column_names, dtype=data_types)


class GenFileReaderWrapper:
    """
    This class wraps FileReaderCarSim and make it suitable for data augmentation pipeline.
    Generally it enables rollover behavior and option to provide shuffled indices for randomized data retrieval.
    """

    def __init__(self, filereader, indices=None, rollover=False):
        """

        :param filereader: Instance of FileReaderCarSim 
        :param indices: list of randomized indices for data shuffling
        :param rollover: If True then __next__ method is generating data infinitely. 
                         Useful for example when used with keras fit_generator
        """
        self.filereader = filereader

        # Use indices if defined, else use indices in sequence 0...N
        if indices is not None:
            self.indices = indices
        else:
            self.indices = range(len(filereader))

        # set start index for indices list
        self.indices_index = 0

        # Rollover means that items are looped over infinitely.
        self.rollover = rollover

        # Some portions of code need to be thread safe so let's define a lock
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __getitem__(self, item):
        """
        This method do sequential or shuffled indexing depending of indices.

        :param item: index
        :return: tuple (x, y)
        """
        # get item index from indices list.
        index = self.indices[item]
        img, y = self.filereader[index]

        # Dimensions need to be expanded from 3D to 4D
        return (np.expand_dims(img, axis=0), y)

    def __len__(self):
        return len(self.filereader)

    def __next__(self):
        """Returns next item. If rollover is True then items are returned infinitely."""
        # Handle indexing in thread-safe manner
        with self.lock:
            if self.indices_index >= len(self.indices):
                # When rollover flag is true we need to check that we don't exceed array bounds
                # and return index back to zero if we do so
                if self.rollover:
                    self.indices_index = 0
                else:
                    # By setting indices_index back to zero we ensure that we don't need
                    # to instantiate this class again for further looping.
                    self.indices_index = 0
                    # Otherwise we need to raise StopIteration exception to stop iteration
                    # when we have iterated through all items
                    raise StopIteration
            # Note that correct indices are handled in __getitem__
            index = self.indices_index
            self.indices_index += 1
        item = self[index]
        return item


class GenFlipLR:
    """
    This generators augment images by flipping it in horizontal direction.
    quantity of items is multiplied by 2.
    """

    def __init__(self, xy_iterable):
        """
        
        :param xy_iterable: generator which will return tuple of (x, y) where 
        x is 4D-tensor. 
        """
        self.i = 0
        self.new_length = len(xy_iterable) * 2
        self.lock = threading.Lock()
        self.iterable = xy_iterable
        self.flipped = False
        self.item = None

    def __len__(self):
        return self.new_length

    def __iter__(self):
        # prepare first item from the iterable to be processed in __next__ method
        return self

    def __next__(self):
        """
        This method returns next left-right flipped image.
        :return: tuple (flipped image, y)
        """
        if self.item is None:
            with self.lock:
                self.item = next(self.iterable)
        # Thread-safe flip flag reversing
        if self.flipped:
            # Thread-safe flip flag reversing
            with self.lock:
                self.flipped = not self.flipped
                self.item = next(self.iterable)
                item = self.item
            return item

        else:
            # Thread-safe flip flag reversing
            with self.lock:
                self.flipped = not self.flipped
                x, y = self.item
            # Squeezing and reshaping is needed. Otherwise image will be upside down.
            shape = x.shape
            x_ = np.fliplr(x.squeeze())
            x_ = np.reshape(x_, shape)
            return x_, -y


class GenRandBrightness():
    """
    This class generates brightness augmentation by a random factor.
    i.e. with factors (-0.1, 0.1) the multiplication factor will be 2
    (original, 
    """

    def __init__(self, xy_iterable, br_range=(0.3, 1), random_state=None):
        """
        :param xy_iterable: generator which will return tuple of (x, y) where 
        x is 4D-tensor. 
        :param br_range: brightness range e.g. (0.5, 0.9)
        """
        self.iterable = xy_iterable
        self.range_start = br_range[0]
        self.range_end = br_range[1]
        random.seed(random_state)

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Generates brightness augmented tensor
        :return: 
        """
        x, y = next(self.iterable)
        x_dtype = x.dtype
        f = random.random() * (self.range_end - self.range_start) + self.range_start
        new_x = (np.multiply(x, f)).astype(dtype=x_dtype)
        return new_x, y


class GenRandHorizontalShift:
    """Generates random horizontal shift"""

    def __init__(self, generator: object, shifts: tuple = (-15, 15), multiplication: int = 1,
                 target_distance: float = 10) -> object:
        """
        
        :param generator: generator which will return tuple of (x, y) where 
        x is 4D-tensor. 
        :param shifts: defines minimum and maximum shift in pixel. Tuple (min, max)
        :param multiplication: defines how many times data is multiplied.
        :param target_distance: Used to define what is the vertical location to 
        where car is steering. You can adjust this in order to define how 
        aggressively car should react to small variations in horizontal shift
        """
        self.generator = generator
        self.shift_min = shifts[0]
        self.shift_max = shifts[1]
        self.multiplication = int(multiplication)
        assert self.multiplication >= 1, "Multiplication factor can't be less than 1"
        self.counter = 0
        self.item = None
        self.lock = threading.Lock()
        self.target_dist = target_distance

    def __len__(self):
        return len(self.generator) * self.multiplication

    def __iter__(self):
        return self

    def __next__(self):
        """Returns next horizontally augmented image. If multiplication is > 1 
        then each image is returned N-times with different shift."""
        with self.lock:
            if (self.item is None) | (self.counter >= self.multiplication):
                self.item = next(self.generator)
                self.counter = 0
            self.counter += 1
        x, y = self.random_hshift(self.item[0], self.item[1])
        return x, y

    def random_hshift(self, x, y):
        """This method applies random shift to tensor and return new 
        corresponding steering angle"""
        img = x[0]
        rows, cols = img.shape[:2]

        # Generates random number to shift
        x_shift = int(random.uniform(self.shift_min, self.shift_max))

        # define transformation matrix
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        M = np.float32([[1, 0, x_shift],
                        [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))

        # calculate new angle based on x_shift
        y = self._calc_new_angle(x_shift=x_shift, angle=y, target_distance=rows * self.target_dist)
        # add dimension (3D --> 4D) and return tensor
        return np.expand_dims(dst, axis=0), y

    def _calc_new_angle(self, x_shift, angle, target_distance=160., shift_div=2):
        """This function calculates a new angle for given X-shift."""
        # Calculate original adjacent and opposite for steering triangle
        adjacent = np.absolute(target_distance)
        opposite = np.tan(np.deg2rad(angle)) * adjacent
        # new_angle = np.rad2deg(np.arctan(vtarget_x / vtarget_y))

        # Add x_shift to triangle's opposite
        # it is good idea to divide x_shift in order to get reasonable angle
        opposite = opposite + x_shift / shift_div

        # Calculate new steering angle, and clip to -25...20 degrees
        new_angle = np.rad2deg(np.arctan(opposite / adjacent))
        new_angle = np.clip(new_angle, -25, 25)

        return new_angle


class GenGaussVerticalShift:
    """Generates gaussian random vertical shift."""

    def __init__(self, generator, mu=0, sigma=5, multiplication=1):
        """
        
        :param generator: generator which will return tuple of (x, y) where 
        x is 4D-tensor. 
        :param mu: mean of random distribution
        :param sigma: standard deviation of random distribution
        :param multiplication: defines how many times data is multiplied.
        """
        self.generator = generator
        self.mu = mu
        self.sigma = sigma
        self.multiplication = int(multiplication)
        assert self.multiplication >= 1, "Multiplication factor can't be less than 1"
        self.counter = 0
        self.item = None
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.generator) * self.multiplication

    def __iter__(self):
        return self

    def __next__(self):
        """Returns next augmented item."""
        with self.lock:
            if (self.item is None) | (self.counter >= self.multiplication):
                self.item = next(self.generator)
                self.counter = 0
            self.counter += 1
        x = self.random_vshift(self.item[0])
        return x, self.item[1]

    def random_vshift(self, x):
        """This method generates random shift."""
        img = x[0]
        rows, cols = img.shape[:2]
        # define transformation matrix
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        y_shift = int(random.gauss(self.mu, self.sigma))
        M = np.float32([[1, 0, 0],
                        [0, 1, y_shift]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        # add dimension (3D --> 4D) and return tensor
        return np.expand_dims(dst, axis=0)


class BatchGenerator:
    """This class is used to generate batches of N-size from the training data.
    This can be used in conjuction with keras.fit_generator."""

    def __init__(self, generator, batch_size=32):
        """
        
        :param generator: generator which will return tuple of (x, y) where 
        x is 4D-tensor. 
        :param batch_size: defines batch size. Generally it should be below 64 for
        good training results.
        """
        self.batch_size = batch_size
        self.i = 0
        self.generator = generator
        self.lock = threading.Lock()

        self.iterators = []

    def __len__(self):
        return len(self.generator)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns next batch of data. This method expects that the underlying generators will generate 
        data items infinitely.
        :return: 
        """
        return self.create_batch()

    def create_batch(self):
        """
        This method creates a batch of data
        :return: 
        """
        # We need to get one item to determine size of np array
        item = next(self.generator)

        # Reserve space for batch
        shape_x = list(item[0].shape)  # Get shape of item
        shape_x[0] = self.batch_size  # adjust N to be batch_size
        x_arr = np.array(np.empty(shape_x), dtype=item[0].dtype)
        y_arr = np.array(np.empty(self.batch_size), dtype=item[1].dtype)

        # insert first item to array
        x_arr[0], y_arr[0] = item

        # Generate batch size of samples
        for i in range(1, self.batch_size):
            item = next(self.generator)
            x_arr[i], y_arr[i] = item
        x, y = x_arr, y_arr  # This is easier to debug than directly returning arrays
        return x, y


def GenTrainTestSplit(filereader, test_size=0.2, random_state=0, rollover=True):
    """This function creates train and test generators.
    :param filereader: FileReaderCarSim instance.
    :param test_size: how big portion of data is used for test set
    :param random_state: set this to integer number if you want consistent 
    randomization of items for each run.
    :param rollover: if True then items are looped over infinitely. Set to true 
    when you use keras.fit_generator
    """
    rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(rs.split(range(len(filereader))))
    train_gen = GenFileReaderWrapper(filereader, indices=train_index, rollover=rollover)
    test_gen = GenFileReaderWrapper(filereader, indices=test_index, rollover=rollover)
    return train_gen, test_gen


class GenPreprocess:
    """
    This class is a image preprocessor generator to be used 
    during model training.
    """
    def __init__(self, generator, cropping=((55, 30), (10, 10)), clipLimit=2.0,
                 tileGridSize=(8, 8), new_range=(-1.0, 1.0)):
        """
        
        :param generator: (x, y) item generator which can supply items infinitely 
        :param cropping: Cropping limits specified as ((top_crop, bottom_crop), (left_crop, right_crop))
        :param clipLimit: CLAHE clip limite
        :param tileGridSize: CLAHE grid size
        :param new_range: Defines value range of output image.
        """
        self.generator = generator
        self.p = ImagePreprocess(cropping=cropping,clipLimit=clipLimit,
                                 tileGridSize=tileGridSize, new_range=new_range)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)

    def __next__(self):
        x, y = next(self.generator)
        img = self.p.apply(x[0])
        x = np.expand_dims(img, axis=0)
        return x, y