import numpy as np
import os
import time

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# import PIL

# img = PIL.Image.open("data_test/test.jpg")
# img.show()

# Force Tensorflow to run on CPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

img_dir = "cat_breed_images"
all_img_paths = []

for file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, file)
    assert(os.path.isfile(img_path))
    all_img_paths.append(img_path)

print(all_img_paths)

model = VGG19(weights='imagenet')
# model = VGG16(weights='imagenet')

cat_breeds = [
    "tabby",
    "tiger_cat",
    "Persian_cat",
    "Siamese_cat",
    "Egyptian_cat",
    "lynx" # Not a cat, but very close to cats
    ]

start = time.perf_counter()
for img_path in all_img_paths:
    img = image.load_img(img_path, target_size=(224, 224))
    # The img_to_array does transpose! From (width, height, channels) to (height, width, channel)!
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print(img_path)
    labels_top = decode_predictions(preds, top=3)[0]
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
    print('Predicted:', labels_top)

    is_a_cat = False
    for label in labels_top:
        if (label[1] in cat_breeds):
            is_a_cat = True
            break

    print("Is it a cat?", is_a_cat)

print(f"Completed Execution in {time.perf_counter() - start} seconds")