import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
#from generator import GenRandHorizontalShift
#from generator import FileReaderCarSim, GenFileReaderWrapper
import numpy as np


def get_session(gpu_fraction=0.33):
    """
    Defines a fraction of GPU memory to allocate.
    https://groups.google.com/forum/#!topic/keras-users/MFUEY9P1sc8

    >>> KTF.set_session(get_session(gpu_fraction=0.25))
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class k_utils:
    """Obsolete version of image preprocessing."""
    @staticmethod
    def image_preprocess(x):
        """
        This is a image preprocessing layer for keras.
        it expects that data value range is 0...255 because there is hardcoded division by 127

        1. Range
        """
        # x = K.cast(x, dtype='float32')
        x = x / 127. - 1
        return x


class ImagePreprocess:
    """New version of image preprocessing"""
    def __init__(self, cropping=((0, 0), (0, 0)), clipLimit=2.0, tileGridSize=(8,8), new_range=(-1.0, 1.0)):
        """
        :param cropping: Cropping limits specified as ((top_crop, bottom_crop), (left_crop, right_crop))
        :param clipLimit CLAHE clip limit
        :param tileGridSize CLAHE grid size
        :param new_range Defines value range of output image.

        >>> p = ImagePreprocess(cropping=((55, 30), (10,10)))
        >>> p.apply(img)
        """

        self.cropping = cropping
        # Use
        self.clahe_obj = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        self.new_range = new_range

    def apply(self, img):
        """
        This method takes RGB image in and run it through processing pipeline.
        
        Pipeline consists of following steps
        
        1. Crop
        2. Color space RGB to HSV
        3. CLAHE
        4. Range
        """
        img = self.crop(img)
        img = self.RGB2HSV(img)
        img = self.clahe(img)
        img = self.im_to_float(img, self.new_range[0], self.new_range[1])
        return img

    def crop(self, x):
        """Crops image.
        :param x: input image
        :return: 
        """
        shape = x.shape
        # [y_top:y_bot, x_left:x_right]
        x = x[self.cropping[0][0]:shape[0] - self.cropping[0][1],
              self.cropping[1][0]:shape[1] - self.cropping[1][1]]
        return x

    @staticmethod
    def RGB2HSV(x):
        """Changes color space from RGB to HSV."""
        return cv2.cvtColor(x, cv2.COLOR_RGB2HSV)

    @staticmethod
    def im_to_float(img, new_min=-1.0, new_max=1.0):
        """Convert image to float32 representation with new given range. 
        Default range is -1...1
        
        :param img: Image to be converted.
        :param new_min: New minimum value.
        :param new_max: NEw maximum value.
        """
        img = cv2.normalize(img.astype(np.float32), None,
                            new_min, new_max,
                            cv2.NORM_MINMAX)
        return img

    def clahe(self, img):
        """Applies CLAHE to image.
        http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        """
        h, s, v = cv2.split(img)
        v = self.clahe_obj.apply(v)  # Apply clahe on value channel
        img = cv2.merge((h, s, v))
        return img


def show_images(driving_log_file):
    """Experimental function to show image from CSV file continuously such like a video stream."""
    i = 0
    font = cv2.QT_FONT_NORMAL
    text = plt.text(x=20, y=40, s="steering angle", fontsize=14, color='red')
    #reader = FileReaderCarSim(driving_log_file, pictures=('center', ))
    #gen = GenFileReaderWrapper(reader,indices=[1,2,3,4,5])
    #gen = GenRandHorizontalShift(gen, (-20, 20), multiplication=1, target_distance=40)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for x, y in gen:
        img = x[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = "index: {}".format(i)
        cv2.putText(img, text, (10, 130), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        text = "angle {:.3f}".format(y)
        cv2.putText(img, text, (10, 150), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', img)
        i += 1
        cv2.waitKey(0)
    cv2.waitKey(5)
    cv2.destroyAllWindows()


def plot_keras_history(history):
    """
    This function plots keras history **loss** and **val_loss** values.
    
    http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    """
    # summarize history for loss and val_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
