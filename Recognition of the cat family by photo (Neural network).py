from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
while True:
  src = input('Path to photo:')
  # Disable scientific notation for clarity
  np.set_printoptions(suppress=True)

  # Load the model
  model = load_model("keras_model.h5", compile=False)

  # Load the labels
  class_names = open("labels.txt", "r").readlines()

  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  # Replace this with the path to your image
  image = Image.open(src).convert("RGB")

  # turn the image into a numpy array
  image_array = np.asarray(image)
  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
  # Predicts the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]


  if confidence_score < 0.6:
    print('Это точно:', class_name[2:], end="")
    print("Confidence Score:", confidence_score)
  else:
    print('Я не уверен, но это:', class_name[2:], end = "")
    print('Confidence Score:', confidence_score)



