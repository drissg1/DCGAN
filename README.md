# Training A DCGAN using tensorflow 2.0

This was a fun project to get some experience with the new tensorflow 2.0 API.  I adapted the basic structure from tensorflows [DCGAN](https://www.tensorflow.org/beta/tutorials/generative/dcgan)

Since I do not have access to an NVIDIA gpu I decided to use Google Colab to run the jupyter notebook that was used in the training of this model.

The original tensorflow post trained on the MNIST dataset and was able to generate realistic looking images after a few epochs. I decided to use the [CelebA Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) for training.

The end goal of this project was to be able to reproduce realistic looking celebrity images and run a poll with my instagram followers. They would vote to see whether the image displayed was of a real celebrity or fake one, a little human vs. discriminator head to head test.  Unfortunately, as you will see the images weren't exactly deceptive.  

# Results
I made a short video that shows the training proccess happening https://github.com/drissg1/DCGAN/blob/master/video/faces.m4v

A few of the generated Images that I cherry picked for their somewhat realistic nature. As you can see they are from from perfect but for only 4 full epochs of training they at least somewhat reasonable features. Obviously symmetry seams to be of no concern for the generator, a common problem for GAN models. 
<div class="row">
  <div class="column">
    <img src="https://github.com/drissg1/DCGAN/blob/master/images/example1%20(11).png"
	  title="Celeb1" width="400" height="400" />
  </div>
  <div class="column">
    <img src="https://github.com/drissg1/DCGAN/blob/master/images/example1%20(12).png" alt="Celeb2"
	  title="Celeb2" width="400" height="400" />  
  </div>
  <div class="column">
    <img src="https://github.com/drissg1/DCGAN/blob/master/images/example1%20(14).png" alt="Celeb1"
	  title="Celeb3" width="400" height="400" />
  </div>
  <div class="column">
    <img src="https://github.com/drissg1/DCGAN/blob/master/images/example1%20(9).png" alt="Celeb1"
	  title="Celeb3" width="400" height="400" />
  </div> 
</div>

## Setup

### Kaggle Dataset and Google Drive API

After some trial and error I was able to authenticate my kaggle token on colab and directly download the dataset onto the local storage provided to me by the wonderful people at Google.

```Python
#Upload kaggle authorization json file
from google.colab import files
files.upload()

#Make directory to store the kaggle json file.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

#Download the Raw Training data zip
!kaggle datasets download -d jessicali9530/celeba-dataset

```
Colab also makes it easy to mount your google drive account to the local file system. I would use my google drive account to store checkpoints of my training as well as save my snapshot images of the training process.

```Python
#Mount Google drive acount for intermitent training 
from google.colab import drive
drive.mount('/content/gdrive')
```
### Load and Preprocess images
After unziping all 200,000+  (218,178) images on the local file system, I created a list to hold the file paths.
```Python
#Gather the names all the jpg images into a python list
from os import listdir
from os.path import isfile, join
fullImgPath = [file_path + f for f in listdir(file_path) if isfile(join(file_path, f))]
fullImgPath =  tf.constant(fullImgPath)
```
I then created a preprocessing function that would scale down the images down and rescale then to the range (-1,1) so that the final layer of my generator being a tanh match the input data domain. 
```Python
def preprocess_image(image,x_crop,y_crop):
  '''Givern an tf.file crop down the image and scale it to be between 0 and 1
  Args: TensorFlow Image, X_crop factor, y_crop factor 
  Returns: Scaled and Center Cropped Image for training
  '''
  
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [x_crop, y_crop])
  image = (image/255) *2 - 1
  return image

def load_and_preprocess_image(path,x_crop=176,y_crop=176):
  '''Given an image file path load that image inot a tensor object and process that image
   Args: Path to Image, x crop factor and y crop factor
   Returns: Scaled and Cropped image ready for training
  '''
  image = tf.io.read_file(path)
  return preprocess_image(image,x_crop,y_crop)
```

### Tensorflow Dataset
Since there are over 200,000 images memory usage could be a problem so I utilized the tensoflow's dataset library to preprocess and load the images into an iterable. This object will only return a selection consisting of "BATCH_SIZE" images for training. Unfortunately this was one of the main issues I encountered while training. It seems as though memory is leaking during training because after about 200 batches of size 64 I reach the memory constraints of Colab. I found a work around but I am not sure what is the cause of the issue.

```Python
#Create the df dataset for training
path_ds = tf.data.Dataset.from_tensor_slices(fullImgPath)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)

#Buffersize and Batch Size
BUFFER_SIZE =  4096
BATCH_SIZE = 64

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
train_dataset = image_ds.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = image_ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
```
## Generator and Discriminator
Next thing was to create the models for the Generator and the Discriminator. The Generator will receive a tensor of shape (100,) and output an image of (176,176,3).

The Discriminator will receive an image of (176,176,3) and output the probability it associates with that image being either real or fake.

```Python

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(11*11*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((11,11, 256)))
    assert model.output_shape == (None, 11, 11, 256) # Note: None is the batch size

    #Deconvolution layers:
    
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 22, 22, 256)
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 22, 22, 256)
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 44, 44, 256)
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 44, 44, 256)
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 88, 88, 128)
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU())


    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 176, 176, 64)
  
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 176, 176, 3)
    
    

    return model
    
    
  
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[176, 176, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
  
    model.add(layers.Activation('sigmoid'))
    
    return model

```
### Losses and Optimizers
Next thing that was needed was to create the loss functions for the generator and the discriminator. 

During the training the Discriminator will receive batches of real images and batches of fake images that were generated by the Generator model. We calculate the cross entropy loss between a tensor of 1s and the images and a tensor of 0s for the fake images created by the generator. The ideal discriminator would be able to predict a 1 for all the real images and a 0 for the images that were generated - causing the respective losses to be zero.  We add up the loss for each to get the total loss for the discriminator. 

```Python
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

```

The Generator loss is fairly similar. We calculate the cross entropy between the scores assigned to the fake images by the Discriminator and a tensor of 1s.  This loss measures how well we have fooled the discriminator.  If we have perfectly fooled the discriminator then it will have assigned a 1 to all the images created by the generator and our cross entropy loss will be 0. 

```Python
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

We need to define two Adam optimizers for both the generator and the discriminator. I debated whether to increase the learning rate for the discriminator so that it will be able to adjust to better images faster but ultimately did not see much of an improvement in training.
```Python
generator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
```
## Training
### Train step

The training step will be repeated Sizeof(dataset)/BatchSize times per epoch. For each step we input a new batch of images, calculate generated images, and then calculate generator and discriminator loss. 

```Python
@tf.function
def train_step(images,step,writer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))     
    
    with writer.as_default():
      tf.summary.scalar('Gen Loss', gen_loss, step=step)
      tf.summary.scalar('Disc Loss', disc_loss, step=step)
```

The @tf.function is new to tensorflow 2.0 and causes the corresponding function to compile into a graph which will allow us to train our model on the gpu provided to us by the colab environment. The Gradientape is used to record all operations done in its scope and then apply the gradients to the parameters using the optimizers we have previously defined. 

## Generating Images

For each epoch we would like to generate images so that we can create a gif of the training process. We would save these images to google drive. A faster method would be to save them locally then after training save the folder to google drive but because of OOM issues experienced with training I didn't want to take any chances. As well the function would run 20 times per run of full dataset. 


```Python

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 4

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim],seed=42)


def generate_and_save_images(model,run, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(8,10))
  
  for i in range(predictions.shape[0]):
      fig.add_subplot(2, 2, i+1)
      plt.imshow((predictions[i]+1)/2)
      plt.axis('off')
  os.makedirs('gdrive/My Drive/gifImgs/run_{}'.format(run),exist_ok=True)
  plt.savefig('gdrive/My Drive/gifImgs/run_{}/image_at_epoch_{}.png'.format(run,epoch),dpi = 300)
  plt.close(fig)
```
## Train Functions
Below is the train functions which should have worked. 
```Python3
@tf.function
def train(dataset, epochs, run, writer):
  for epoch in range(epochs):
    for step,image_batch in enumerate(dataset):
      train_step(image_batch,step,writer)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             run,
                             epoch,
                             seed)
    
    print('Completed epoch :{}'.format(epoch))
    checkpoint.save(file_prefix = checkpoint_prefix)
   

  # save check point after final run:
  checkpoint.save(file_prefix = checkpoint_prefix)
  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           run,
                           epoch,
                           seed)
```
Since the dataset object I created earlier is an iterable there should not be any memory issues during training because only the images that are being worked on will be loaded into memory. However, I tried to debug this function and was not able to, perhaps this is a bug of tensorflow 2.0 beta- more likely I don't understand the correct way to iterate through the dataset object. 

Anyways below is the hack I ended up using. I found that I could run through a batch of 10,000 images before without having any issues with colab's memory usage. So I decided that I would loop over 200,000 images in groups of 10,000 storing them in a new dataset and then run through that small iterable. 

```Python

def train_iterative(fullImgPath, epochs, run, writer):
  for epoch in range(epochs):
    for i in range(20):
      #Create the df dataset for training
      path_ds = tf.data.Dataset.from_tensor_slices(fullImgPath[i*10000:(i+1)*10000])
      image_ds = path_ds.map(load_and_preprocess_image)

      # Setting a shuffle buffer size as large as the dataset ensures that the data is
      # completely shuffled.
      train_dataset = image_ds.batch(BATCH_SIZE)
      
   
      for step,image_batch in enumerate(train_dataset):
        train_step(image_batch,step,writer)

      # Produce images for the GIF as we go
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                               run,
                               i,
                               seed)
      print('Completed step :{}'.format(i))
      checkpoint.save(file_prefix = checkpoint_prefix)

    print('Completed epoch: {}'.format(epoch))
  ```



