import os

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL_DIR = "vgg_serving"
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print(export_path)

model = VGG19(weights='imagenet')

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=False
)

enable_load_test = True

# Load Model Test
if enable_load_test:
    import numpy as np

    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg19 import preprocess_input

    print("Do load test.")

    loaded_model = tf.keras.models.load_model(export_path)
    img_path = "data_test/test.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    assert np.allclose(model.predict(x), loaded_model.predict(x))

