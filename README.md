## Training A DCGAN using tensorflow 2.0

This was a fun project to get some expereince with the new tensorflow 2.0 API.  I adapated the basic structure from tensorflows [DCGAN](https://www.tensorflow.org/beta/tutorials/generative/dcgan)

Since I do not have access to an NVIDIA gpu I decided to use Google Colab to run the jupyter notebook that was used in the training of this model.

The original tensorflow post trained on the MNIST dataset and was able to generate realistic looking images after a few epochs. I decided to use the [CelebA Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) for training.

The end goal of this project was to be able to reproduce realistic looking celebrity images and run a poll with my instagram followers. They would vote to see whether the image displayed was of a real celebrity or fake one, a little human vs. discriminator head to head test.  Unfortunately, as you will see the images weren't exactly decepitive.  

### Kaggle Dataset and Google Drive API

After some trial and error I was able to authenticate my kaggle token on colab and directly download the datset onto the local storage provided to me by the wonderful people at Google.

```markdown
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

```
#Mount Google drive acount for intermitent training 
from google.colab import drive
drive.mount('/content/gdrive')
```
### Load and Preprocess images
After unziping all 200,000+  (218,178) images on the local file system, I created a list to hold the file paths.
```
#Gather the names all the jpg images into a python list
from os import listdir
from os.path import isfile, join
fullImgPath = [file_path + f for f in listdir(file_path) if isfile(join(file_path, f))]
fullImgPath =  tf.constant(fullImgPath)
```
I then created a preprocessing function that would scale down the images down and rescale then to the range (-1,1) so that the final layer of my generator being a tanh match the input data domain. 
```
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

###

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/drissg1/DCGAN/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
