# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./illustrations/histogram_center_img.png "Steering angle distribution on center images"
[image2]: ./illustrations/model_06.png "Model 06"
[image3]: ./illustrations/center_2017_04_04_21_53_05_533.jpg "Center lane driving"
[image4]: ./illustrations/center_2017_04_04_21_52_29_824.jpg "Cornering"
[image5]: ./illustrations/recovery_driving.png "Recovery Driving"
[image6]: ./illustrations/random_brightness_20170410.png "Random Brightness Augmentation"
[image7]: ./illustrations/fliplr_20170410.png "Left-Right Flip Augmentation"
[image8]: ./illustrations/vertical_shift_20170410.png "Vertical Shift Augmentation"
[image9]: ./illustrations/horizontal_shift_20170410.png "Horizontal Shift Augmentation"
[image10]: ./illustrations/augment_pipeline_20170411.png "Image Run Through Augmentation Pipeline"
[image11]: ./illustrations/training_plot_model_06_e50_lr0_001.png "Training loss and val_loss plotted"

[//]: # (Video References)
[video1]: ./video.mp4 "Final result: Car driving autonomously"

[//]: # (Article References)
[1]: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
[2]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
[3]: https://wiki.python.org/moin/Generators

## Rubric Points

**Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) 
individually and describe how I addressed each point in my implementation.**  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Code
  - model.py containing the script to create model
  - drive.py for driving the car in autonomous mode
  - model.h5 containing a trained convolution neural network 
  - train.py used for training the model
  - generator.py contains all the data generators needed for training and data augmentation
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be 
driven autonomously around the track by executing

```sh
python drive.py model.h5
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, 
and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of MaxPooling layer to reduce input image size 
from (75, 300, 3) to (75, 75, 3). After MAxPooling there is one convolutional 
layer with 2x2 sub-sampling and 3x3 kernel to further reduce image size down to
(37, 37, 4) 

Then there are nine convolution layers with kernel size 3x3 which are 
reducing image size 19x19 to and increasing depth to 24. These layers are 
converting features in images to more high level features. 
 
Last 2 convolution layers are reducing image dimensions from 19x19 to 4x15 
and keeping depth at 24. Image size on these two layers is relative wide 
compared to earlier layers. That is to keep needed parameters small and still
extract useful features about the road curvature.

Finally there are 5 fully connected layers which are processing the output from 
convolutional layers. Last layer is linear output layer which outputs steering 
angle as a floating point number in range -1...1.

All layers are using ELU activation except last 2 layers. Output layer uses 
linear activation and 1 layer before output is sigmoid layer which is creating
"clean" activation for linear output layer.

Reason why I'm using ELU activation is that it reduces needed for batch 
normalization and therefore i can make simpler model.

model is defined `model.py` lines 14-84

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. In 
convolution layers i'm using 0.1 dropout rate and  FC1-FC4 are using 0.5 
dropout rate. Last 3 layers are not using dropout.

I was using small data sets to evaluate models in order to understand fast is 
model working or not. Mostly I paid attention to over-fitting and if model was 
overfitting data then it was good sign in this initial step because then i had 
some certainty that model can memorize track. Also the over-fitting problem will 
be tackled by using more training data and data augmenting

#### 3. Model parameter tuning

I'm using adam optimizer with learning rate 0.001 and batch size 32. After 
several times of trial and error I came into conclusion that those are working 
well. I used **mean-squared-error** as a loss function. For further development 
there should be more systematic way of tuning hyperparameters. Good candidates 
for parameter tuning would be K-fold cross-validation and grid search.

In beginning i tried also batch size of 128...256 and noticed that with those 
sizes training is not performing well. After dropping batch size below 64 i begin
to get much better results.

Below is how I defined optimizer and loss function.

````python
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
model.compile(optimizer=adam, loss='mean_squared_error' )
````


#### 4. Appropriate training data

Training data should be cover enough different type of scenarios such as 
driving on center of road, corners, recovery from side of the track and off 
track.
 
By experimenting models and training i noticed that there is need for more 
training data from certain locations of the track. These include bridge and 
some specific corners and areas with tree shadows.

Model was fine tuned after adding more training data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first try out 
available models such as VGG16 and if it doesn't work then after that try to 
create own model.

First i was using VGG16 model with custom output layer which consist of few 
fully connected layers. Anyhow i didn't get this to work for some reason. 

Then the next step was to try out [NVidia's model][2]. It turned out that this 
model was working better, but still not well enough. 
(As a side note i can say that it could be because of insufficient training 
data or hyper-parameters)

Anyhow.. i came up with new architecture which contains 13 convolutional layers 
and 5 fully connected layers.

I split data into training and validation sets in order study model 
performance better. It was quite interesting that my model gave quite good 
readings for **loss** and **validation loss** even though car couldn't stay on 
road. Those number were in vicinity of 0.03. I'm not exactly sure is it because 
of insufficient training data or some other reason. 

In beginning i was using batch size 128 or 256 which caused plenty of headache 
and model didn't train well and didn't generalize well on unseen data. Then 
after studying the issue i dropped batch size to 64 and even to 32. Result was 
big improvement on model training behavior and now car actually begin to stay 
on track. Again need to mention that it is important to study effect of 
hyper-parameters when developing or testing the models.

Overfitting was tackled by dropout layers in model and also by augmenting 
training data. 
 
I was testing model in simulator quite often during development. In later parts 
of the development when car was able to drive for a while it was very useful to 
see problematic areas in simulator and then add more training data about those 
problematic places.

From below histogram you can see that how vast majority of steering angles 
are near zero so it explains on some level why model sometimes have problems 
in tight corners.

![steering angle histogram][image1]

At the end of the process, the vehicle is able to drive autonomously around 
the track without leaving the road. There were only one or 2 places which 
should be improved to be perfect.

#### 2. Final Model Architecture

The final model architecture consists totally of 18 layers (MaxPool, Conv2D and Fully connected)
In top of the model we have MAxPool to reduce image size to 75x75 and then there are totally 12 convolutional layers.
And last here are 5 fully connected layers.

Here is a visualization of the architecture.
 

![Model_06][image2]

#### 3. Creation of the Training Set & Training Process

My initial target was to do all the training data gathering on track one and 
use data augmentation techniques to help model to generalize so that it could 
drive also on track two. That is quite a big challenge as those tracks are 
totally different. Track one is kind of a race track on flat ground where as 
track two is more like narrow mountain road.

So let's begin..

First step is to capture data of good driving behavior on center of the road.
From this data model is able to learn how it should drive on center of the road.


To capture good driving behavior, I first recorded two laps on track one 
using center lane driving. Here is an example image of center lane driving:

![Center lane driving][image3]

Regarding to histogram of steering angles earlier in this document I came into 
conclusion that straight portions of the road are over presented in training 
data so then i decide to collect training data from corner driving. I collected 
more data about corners by driving on track one lap into both directions and 
recording driving on each corner.

![Cornering][image4]

Now on this point of training data collection we should have enough data to train
car to drive on center of the road on straight road and in corners.

But there is problem that what car should do when it is off center? No problems! 
We can also teach that to the model. This will be done by collecting so called 
recovery driving where we drive car on side of the road, turn recording on and 
drive it back to center of the lane. And repeating this step several times. I 
also added some training data from situations when car is just about to go off 
center. Later i noticed that steering angles have to be kept quite small also in 
recovery mode, because when car is driving fast it is bad idea to make sudden 
turns.

![Recovery Driving][image5]

Because we are using only one track we need to figure out how to augment data so 
that car could drive on track #2.

After collecting training data i had totally 47355 samples (center, left and right cameras)
This is easily multiplied by data augmentation and the augmented set includes 568k samples

##### Data augmentation

I'm defining data augmentation for model training in following function which is 
located `train.py` 14-22. There i can define easily what augmentation generators 
are used.

```python
def augment_pipeline(generator):
    gen = generator
    gen = GenPreprocess(gen)  # Add preprocessing here as it is computationally intensive, brightness generator have to be before it, but later steps are multiplying data and therefore reducing CPU burden
    gen = GenRandBrightness(gen)
    gen = GenFlipLR(gen)
    gen = GenGaussVerticalShift(gen, mu=0, sigma=8, multiplication=2)
    gen = GenRandHorizontalShift(gen, (-10, 10), multiplication=3, target_distance=20)
    
    return gen
```

Before the data augmentation there is image pre-processing step which:
* changes colorspace from RGB to HSV
* crop image to size (75, 300, 3)
* applies adaptive histogram correction. 

First augmentation generator which i implemented was random brightness generator.
This generator changed original image brightness randomly in order to train model
for variable image brightness.

Below image shows 4 different images where random brightness is applied.

![Random Brightness Augmentation][image6]


Second implemented data augmentation generator was left-right flip. Which flips 
the image from left to right. This together with negating the steering angle 
gives us double amount of training data

![Left-Right Flip Augmentation][image7]


Third augmentation generator was vertical shift which shifts image randon number
of pixels in vertical direction. I suppose this should generally reduce 
overfitting small amounts and then on the other hand it prepares model to handle
roads with slope.

![Vertical Shift Augmentation][image8]


Fourth augmentation generator was horizontal shift. This proved to be really the most 
important data augmentation generator because it can generate small vertical shifts
together with steering angle compensation. This really helps model to better 
learn to steer towards center of the road.

![Horizontal Shift Augmentation][image9]


After the whole augmentation pipeline we have several different versions of 
the original image.

![Image After Data Augmenting Pipeline][image10]


#### Train Test Spit + Shuffle

Test data is splitted and shuffled into train and test sets. Split ratio was 0.05.
This is done in `train.py` and below is code snipped how i am using train test split generator.


````python
    # Establish CSV file reader and basic Generators
    train_valid_reader = FileReaderCarSim(driving_log_file)
    # Split into training and validation sets
    train_gen, test_gen = GenTrainTestSplit(train_valid_reader,
                                            test_size=0.05, rollover=True)
````

#### Training

I had problem that i couldn't test model in simulator so when i was training 
the model because training reserved all of GPU memory and model was saved after 
the training was completed. 

I solved this problem by using checkpointer and limiting GPU memory use per session.

I created check pointer which saved model after each epoch. Number of epoch and 
validation loss was appended into filename in order to easily get basic information
of the model. This was useful because then i was able to test model also when training
was ongoing and no need to wait for results for several hours.

````python 
checkpointer = ModelCheckpoint(filepath=filename, verbose=1,
                               monitor='val_loss', save_best_only=False)
````


Other useful thing which helped to experiment multiple models on parallel 
was to limit GPU memory usage. This also allowed me to test model with simulator 
when i was still training model

````python
sess = get_session(gpu_fraction=0.3)
KTF.set_session(sess)
````

I mention generators several times already and now i am going to explain why i'm
using those.

Inherent problem is you need a lot of training data and a lot of date means that
 you need plenty of RAM to handle it. Even with this small project i was 
easily able to use all of my computer's RAM (64GB). As a curiosity my latest 
model is using roughly 750k samples and each sample is about 140kB so total 
amount of RAM needed to keep all data in RAM is ~100GB. Doable but doesn't make 
sense.

So now we got to talk about generators which will save my budged by providing 
way to keep only ony sample in memory. OK.. in practice there were perhaps few 
hundred samples including all the buffers. More information about generators is 
 available in [python wiki][3]

Generators were able to generate augmented data in parallel with 
training operation. So this did not cause slowdown to training. Below code snippet
shows how i was using training and validation generators with kera's fit_generator.

````python
hist = model.fit_generator(train_gen, samples_per_epoch=len(train_gen), nb_epoch=nb_epoch,
                           verbose=verbose,
                           validation_data=test_gen, nb_val_samples=len(test_gen),
                           max_q_size=batch_size*2, nb_worker=3, pickle_safe=True,
                           callbacks=[checkpointer],
                           initial_epoch=0)
````

Final test was to test whether car can drive autonomously on track. Most of the
 time car was behaving quite correctly with speed 15mph. I observed slight 
 problems when car was approaching edge of the road which caused sometimes car 
 to steer steeply. That behavior will cause problems in higher speeds. I tested 
 on 20mph and car begin to oscillate between lane edges and finally drove off the road.
 Anyhow.. slower speeds are still ok.
 
Final model was training for 50 epochs and from the training and validation loss 
plot we can see that model is converging quite nicely. 
Final metrics are `loss: 0.0217 - val_loss: 0.0149   `

![Model 6 training and validation loss][image11]

 
 Then next step is to find out how model can drive car around the track. Below 
 is result video which was driven with model trained for 50 epochs. In video car 
 is driving close to left edge of the road, it could be caused by the training 
 data where i'm also driving bit closer to left edge.
 
![Result Video: Car Driving Autonomously][video1]

At this moment car can't drive on track #2. Either I need to collect training 
data from that particular track or try to augment data collected from track #1 
in different ways. This will be left for further study.

