# =============================================================================
# CARVANA IMAGE SEGMENTATION WITH U-NET
# shamelessly copied form tutorial
# https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb
# =============================================================================

import os
import glob
import zipfile
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} ) 
sess = tf.Session(config=config) 
K.set_session(sess)
#%%
# =============================================================================
# COLLECTING THE DATA
# =============================================================================

def get_kaggle_credentials():
  token_dir = os.path.join(os.path.expanduser("~"),".kaggle")
  token_file = os.path.join(token_dir, "kaggle.json")
  if not os.path.isdir(token_dir):
    os.mkdir(token_dir)
  try:
    with open(token_file,'r') as f:
      pass
  except IOError as no_file:
    try:
      from google.colab import files
    except ImportError:
      raise no_file
    
    uploaded = files.upload()
    
    if "kaggle.json" not in uploaded:
      raise ValueError("You need an API key! see: "
                       "https://github.com/Kaggle/kaggle-api#api-credentials")
    with open(token_file, "wb") as f:
      f.write(uploaded["kaggle.json"])
    os.chmod(token_file, 600)

get_kaggle_credentials()

competition_name = 'carvana-image-masking-challenge'

    
data_dir = 'E:\carvana_dataset'
img_dir = os.path.join(data_dir, "train")
label_dir = os.path.join(data_dir, "train_masks")

df_train = pd.read_csv(os.path.join(data_dir, "train_masks.csv"))
ids_train = df_train['img'].map(lambda s: s.split('.')[0])


x_train_filenames = []
y_train_filenames = []

for img_id in ids_train:
    x_train_filenames.append(os.path.join(img_dir, "{}.jpg".format(img_id)))
    y_train_filenames.append(os.path.join(label_dir, "{}_mask.gif".format(img_id)))
    

#splitting data and test
x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
train_test_split(x_train_filenames, y_train_filenames, test_size = 0.2, random_state = 42)

num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)

print("numero esempi training: {}".format(num_train_examples))
print("numero esempi test: {}".format(num_val_examples))

#%%
# =============================================================================
# VISUALIZING SOME PICS
# =============================================================================
display_num = 5

r_choices = np.random.choice(num_train_examples, display_num)

plt.figure(figsize=(10, 15))
for i in range(0, display_num * 2, 2):
  img_num = r_choices[i // 2]
  x_pathname = x_train_filenames[img_num]
  y_pathname = y_train_filenames[img_num]
  
  plt.subplot(display_num, 2, i + 1)
  plt.imshow(mpimg.imread(x_pathname))
  plt.title("Immagine Originale")
  
  example_labels = Image.open(y_pathname)
  label_vals = np.unique(example_labels)
  
  plt.subplot(display_num, 2, i + 2)
  plt.imshow(example_labels)
  plt.title("Masking")  
  
plt.suptitle("Immagini Carvana Dataset e Masking")
plt.show()


#%%
# =============================================================================
# PREPROCESSING
# =============================================================================

# keep image size divisible by 32
img_shape = (512, 512, 3)

batch_size = 3
epochs = 15

"""
SOME PLANNING

1 > we need to read bytes from image example and relative mask
2 > turn that into an actual image
3 > data augmentation:
    > resize to img_shape
    > hue_delta
    > horizontal_flip
    > width_shift_range height_shift_range
    > rescale
4 > shuffle!

"""

def _process_pathnames(fname, label_path):
    
    # image
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels = 3)
    
    # label image
    label_img_str = tf.read_file(label_path)
    
    #gif images have frames!
    label_img = tf.image.decode_gif(label_img_str)[0]
    # and also channels
    label_img = label_img[:,:,0]
    
    # not sure why i need to expand
    label_img = tf.expand_dims(label_img, axis = -1)
    
    return img, label_img


# SHIFTING 
def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([],
                                                  -width_shift_range*img_shape[1],
                                                  width_shift_range*img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                                  -height_shift_range*img_shape[0],
                                                  height_shift_range*img_shape[0])
            
        output_img = tfcontrib.image.translate(output_img,
                                               [width_shift_range, height_shift_range])
        label_img = tfcontrib.image.translate(label_img,
                                              [width_shift_range, height_shift_range])
    
    return output_img, label_img


# FLIPPING
def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
        
    return tr_img, label_img
    

def _argument(img,
              label_img,
              resize = None,
              scale = 1,
              hue_delta = 0,
              horizontal_flip = False,
              width_shift_range = 0,
              height_shift_range = 0):
    if resize is not None:
        label_img = tf.image.resize_images(label_img, resize)
        img = tf.image.resize_images(img, resize)
        
    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)
    
    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
    
    
    label_img = tf.to_float(label_img)*scale
    img = tf.to_float(img)*scale
    
    return img, label_img


def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn = functools.partial(_argument),
                         threads = 16,
                         batch_size = batch_size,
                         shuffle = True):
    num_x = len(filenames)
    
    # create dataset from filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # multithread mapping the preprocessing function to every element in the dataset
    
    # intanto carico le immagini
    dataset = dataset.map(_process_pathnames, num_parallel_calls = threads)
    
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images should be same size" #why
    
    # poi le preprocesso
    dataset = dataset.map(preproc_fn, num_parallel_calls = threads)
    
    if shuffle:
        dataset = dataset.shuffle(num_x)
        
    dataset = dataset.repeat().batch(batch_size)
    
    
    return dataset

#%%
# =============================================================================
# TRAIN AND VALIDATION DATASET SETUP
# =============================================================================

tr_cfg = {
        'resize': [img_shape[0], img_shape[1]],
        'scale': 1 / 255.,
        'hue_delta': 0.1,
        'horizontal_flip': True,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1
        }
tr_preprocessing_fn = functools.partial(_argument, **tr_cfg)

val_cfg = {
        'resize': [img_shape[0], img_shape[1]],
        'scale': 1/255.,
        }
val_preprocessing_fn = functools.partial(_argument, **val_cfg)

train_ds = get_baseline_dataset(x_train_filenames,
                                y_train_filenames,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)

val_ds = get_baseline_dataset(x_val_filenames,
                              y_val_filenames,
                              preproc_fn=val_preprocessing_fn,
                              batch_size=batch_size)


# MOMENT OF TRUTH FOR THE IMAGE MAGIC MACHINE
temp_ds = get_baseline_dataset(x_train_filenames, 
                               y_train_filenames,
                               preproc_fn=tr_preprocessing_fn,
                               batch_size=1,
                               shuffle=False)
# Let's examine some of these augmented images
data_aug_iter = temp_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()
with tf.Session() as sess: 
  batch_of_imgs, label = sess.run(next_element)

  # Running next element in our graph will produce a batch of images
  plt.figure(figsize=(10, 10))
  img = batch_of_imgs[0]

  plt.subplot(1, 2, 1)
  plt.imshow(img)

  plt.subplot(1, 2, 2)
  plt.imshow(label[0, :, :, 0])
  plt.show()
  
  
  
# =============================================================================
# MODEL
# =============================================================================

def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  return decoder

# =============================================================================
# NET DRAWING
# =============================================================================

inputs = layers.Input(shape=img_shape)
# 256

encoder0_pool, encoder0 = encoder_block(inputs, 32)
# 128

encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
# 64

encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
# 32

encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
# 16

encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
# 8

center = conv_block(encoder4_pool, 1024)
# center

decoder4 = decoder_block(center, encoder4, 512)
# 16

decoder3 = decoder_block(decoder4, encoder3, 256)
# 32

decoder2 = decoder_block(decoder3, encoder2, 128)
# 64

decoder1 = decoder_block(decoder2, encoder1, 64)
# 128

decoder0 = decoder_block(decoder1, encoder0, 32)
# 256

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

model = models.Model(inputs=[inputs], outputs=[outputs])
#%%
# =============================================================================
# METRICS AND LOSS
# =============================================================================
    
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss



#%%
    # COMPILE MODEL
    
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])

model.summary()





#%%
# =============================================================================
#  HIC SUNT LEONES
# =============================================================================

save_model_path = 'E:\carvana_dataset\model_weights\weights.hdf5'

cp = tf.keras.callbacks.ModelCheckpoint(filepath = save_model_path, monitor = 'val_dice_loss', save_best_only = True, verbose = 1)

history = model.fit(train_ds,
                    steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                    epochs=epochs,
                    validation_data=val_ds,
                    validation_steps=int(np.ceil(num_val_examples / float(batch_size))),
                    callbacks = [cp])

dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, dice, label='Training Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()


#%%
## Alternatively, load the weights directly: model.load_weights(save_model_path)
#model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
#                                                           'dice_loss': dice_loss})


# Let's visualize some of the outputs 
data_aug_iter = val_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()

# Running next element in our graph will produce a batch of images
plt.figure(figsize=(10, 20))
for i in range(5):
  batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
  img = batch_of_imgs[0]
  predicted_label = model.predict(batch_of_imgs)[0]

  plt.subplot(5, 3, 3 * i + 1)
  plt.imshow(img)
  plt.title("Input image")
  
  plt.subplot(5, 3, 3 * i + 2)
  plt.imshow(label[0, :, :, 0])
  plt.title("Actual Mask")
  plt.subplot(5, 3, 3 * i + 3)
  plt.imshow(predicted_label[:, :, 0])
  plt.title("Predicted Mask")
plt.suptitle("Examples of Input Image, Label, and Prediction")
plt.show()
