# Importare:
# !pip install -q imageio
# !pip install -q opencv-python
# !pip install "tensorflow>=2.5.0"
# !pip install --upgrade tensorflow-hub
# !pip install ipython

# no display tf warnings
from absl import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity(logging.ERROR)
# no display tf warnings
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import random
# chart libraries
import plotly.express as px
# local modules
import handlingData
import dataset


def predict(loaded_video):
    model_input = tf.constant(loaded_video, dtype=tf.float32)[tf.newaxis, ...]
    model = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(model)
    print("Top 3 actions:")
    for i in np.argsort(probabilities)[::-1][:3]:
        print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")

    # Creating Line Chart
    fig = px.line(x=probabilities, y=labels, title="Results")
    fig.show()


if __name__ == "__main__":
    ucf_videos = handlingData.list_videos()
    dataset.set_categories(ucf_videos)
    i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']
    labels = dataset.load_kinetics_labels()

    choice = 1
    j = 3
    while j != 0:
        while True:
            print("-------------------------------------------------------")
            # print("(1)Import random video, (2)Open webcam or (0)To exit")
            j = int(input("(1)Import random video, (2)Open webcam or (0)To exit \n"))
            if j == 1:
                video_path = handlingData.fetch_video(random.choice(handlingData.random_list))
                video = handlingData.load_video(video_path)
                break
            elif j == 2:
                while choice == 1:
                    print("You got 10 seconds to perform an action")
                    video = handlingData.load_video(0, 400)
                    choice = int(input("(1)Try again?, (2)Go ahead \n"))
                break
            elif j == 0:
                break
        if j != 0:
            video.shape
            predict(video)
