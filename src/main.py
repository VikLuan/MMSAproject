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

import numpy as np

def predict(loaded_video):
    model_input = tf.constant(loaded_video, dtype=tf.float32)[tf.newaxis, ...]
    model = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(model)
    print("Top 5 actions:")
    for i in np.argsort(probabilities)[::-1][:5]:
      print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")

if __name__ == "__main__":
    video_path = handlingData.fetch_video("v_LongJump_g01_c01.avi")
    video = handlingData.load_video(video_path)
    video1 = handlingData.load_video(video_path)[:100]
    video.shape
    ucf_videos = handlingData.list_videos()
    dataset.set_categories()
    i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']
    labels = dataset.load_kinetics_labels()
    predict(video)
