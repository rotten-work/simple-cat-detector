import numpy as np
import json
import requests

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions

img_path = "data_test/test.jpg"
MODEL_DIR = "vgg_serving"

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

# data = json.dumps({"signature_name": "serving_default", "instances": x.tolist()})
# headers = {"content-type": "application/json"}
# json_response = requests.post("http://localhost:8501/v1/models/vgg_serving:predict", data=data, headers=headers)

data = json.dumps({"instances": x.tolist()})
json_response = requests.post(f"http://localhost:8501/v1/models/{MODEL_DIR}:predict", data=data)

prediction_list = json.loads(json_response.text)['predictions']
# converting list to array
prediction_array = np.asarray(prediction_list)
labels_top = decode_predictions(prediction_array, top=3)[0]
print("labels_top:", labels_top)

test = True

if test:
    import os
    from tensorflow.keras.applications.vgg19 import VGG19

    # GPU results might be different from CPU results
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("Do comparison test.")
    model = VGG19(weights='imagenet')
    preds = model.predict(x)
    labels_top_1 = decode_predictions(preds, top=3)[0]
    print("labels_top_1:", labels_top_1)

    assert np.allclose(preds, prediction_array)

