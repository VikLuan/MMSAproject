# Importare:
# !pip install -q imageio
# !pip install -q opencv-python
# !pip install "tensorflow>=2.5.0"
# !pip install --upgrade tensorflow-hub
# !pip install ipython

# TensorFlow and TF-Hub modules.
from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import handlingData
import dataset

logging.set_verbosity(logging.ERROR)

# Some modules to help with reading the UCF101 dataset.

import numpy as np

#Step 5: Fetch a random video

video_path = handlingData.fetch_ucf_video("v_LongJump_g01_c01.avi")
sample_video = handlingData.load_video(video_path)
sample_video1 = handlingData.load_video(video_path)[:100]
sample_video.shape

#Step 6: Predict from the video

i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

def predict(sample_video):
  # Add a batch axis to the to the sample video.
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

  logits = i3d(model_input)['default'][0]
  probabilities = tf.nn.softmax(logits)
  labels = dataset.getLabels()

  print("Top 5 actions:")
  for i in np.argsort(probabilities)[::-1][:5]:
    print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")

predict(sample_video)
