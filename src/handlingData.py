import re
import os
import tempfile
import ssl
import cv2

import numpy as np
import imageio
from tensorflow_docs.vis import embed

from urllib import request  # requires python3

UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
unverified_context = ssl._create_unverified_context()
random_list = ["v_BrushingTeeth_g01_c01.avi", "v_SalsaSpin_g01_c01.avi", "v_BabyCrawling_g01_c01.avi",
               "v_PlayingCello_g01_c01.avi", "v_LongJump_g01_c01.avi"]


def list_videos():
    global _VIDEO_LIST
    if not _VIDEO_LIST:
        index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode("utf-8")
        videos = re.findall("(v_[\w_]+\.avi)", index)
        _VIDEO_LIST = sorted(set(videos))
    return list(_VIDEO_LIST)


def fetch_video(video):
    cache_path = os.path.join(_CACHE_DIR, video)
    if not os.path.exists(cache_path):
        urlpath = request.urljoin(UCF_ROOT, video)
        print("Fetching %s => %s" % (urlpath, cache_path))
        data = request.urlopen(urlpath, context=unverified_context).read()
        open(cache_path, "wb").write(data)
    return cache_path


def crop_center(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        if path == 0:

            while len(frames) != max_frames:
                ret, frame = cap.read()
                cv2.imshow("Webcam", frame)
                frame = crop_center(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                if not ret:
                    break
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                if len(frames) == max_frames:
                    break
    finally:
        cap.release()
    return np.array(frames) / 255.0
