import json
import math
import os
import queue
import random
import shutil
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from decord import VideoReader, DECORDError
from moviepy.editor import VideoClip, VideoFileClip
import imageio
import av

class VideoLoadException(Exception):
    pass


def get_video_length(video_path, use_meta=False):
    try:
        video = cv2.VideoCapture(video_path)
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        vreader = VideoReader(video_path)
        video_length = len(vreader)

        if use_meta:
            return int(min(video_length, count))

        if is_video_ok(video_path):
            return int(min(video_length, count))
        
        count2 = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            else:
                count2 += 1
        
        if (video_length != count2) or (count != count2):
            print("warning.....  {} video length not same, cv2: {} decord: {} cv2_read: {}".format(video_path, count, video_length, count2))

        video_length = int(min(video_length, count, count2))
    except Exception as e:
        raise VideoLoadException()

    return int(min(video_length, count))


def get_video_size(video_path):
    # Try using Decord
    try:
        vreader = VideoReader(video_path)
        frame = vreader[0]
        video_height, video_width = frame.shape[:2]
        return video_height, video_width
    except Exception as e:
        print(f"[Warning] Decord failed to read video {video_path}: {e}")

    # Try using OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("OpenCV cannot open the video file.")
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return video_height, video_width
    except Exception as e:
        print(f"[Warning] OpenCV failed to read video {video_path}: {e}")

    # Try using AV
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        video_width = stream.width
        video_height = stream.height
        container.close()
        return video_height, video_width
    except Exception as e:
        print(f"[Warning] PyAV failed to read video {video_path}: {e}")

    # If all methods fail, raise an exception
    raise VideoLoadException(f"Failed to get video size for {video_path} using any method.")

#def get_video_size(video_path):
#    try:
#        vreader = VideoReader(video_path)
#        video_height = vreader[0].shape[0]
#        video_width = vreader[0].shape[1]
##    except Exception as e:
##        raise VideoLoadException()
## test version
#    except DECORDError:
#         # Fallback to imageio if decord fails
#        reader = imageio.get_reader(video_path, 'ffmpeg')
#        meta = reader.get_meta_data()
#        video_height = meta['size'][0]
#        video_width = meta['size'][1]
#        #return meta['size'][0], meta['size'][1]  # (height, width)
#
#    return video_height, video_width


def crc64_check(video_path):
    try:
        import tos
        crc64 = tos.utils.Crc64()
        with open(video_path, "rb") as f:
            crc64.update(f.read())
        print("video_path {} crc64: {}".format(video_path, crc64.crc))
    except Exception as e:
        traceback.print_exc()


def video_reader_check(video_path):
    try:
        video_fps = VideoFileClip(video_path, audio=False).fps
        video_height, video_width = get_video_size(video_path)
        clip_length = get_video_length(video_path, use_meta=True)
        print(video_path, video_fps, video_width, video_height, "video_meta_length:", clip_length)
    except Exception as e:
        traceback.print_exc()
        return "fail"

    return "ok"


def is_audio_ok(video_path):
    t1 = time.time()
    import subprocess
    ffmpeg_bin = "/home/model-server/dev/data_platform/mytools/ffmpeg"
    command = f"{ffmpeg_bin} -v error -i {video_path} -t 20 -f null -"
    result = subprocess.run(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        shell=True
    )

    result1 = result.stdout.strip()
    result2 = result.stderr.strip()
    if "[aac" in result2:
        print(result2)
        return False
    print("is_audio_ok", time.time() - t1)
    return True


def is_video_ok(video_path):
    t1 = time.time()
    import subprocess
    ffmpeg_bin = "/home/model-server/dev/data_platform/mytools/ffmpeg"
    command = f"{ffmpeg_bin} -v error -i {video_path} -t 20 -f null -"
    result = subprocess.run(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        shell=True
    )

    result1 = result.stdout.strip()
    result2 = result.stderr.strip()
    if len(result2) > 0:
        print(result2)
        return False
    print("is_video_ok", time.time() - t1)
    return True



class VideoReaderCV2:
    def __init__(self, video_path, video_length=None, dst_size=None, queue_size=4):
        self.video_path = video_path
        self.video_length = video_length
        self.dst_size = dst_size
        self.queue_size = queue_size
        self.video = None
        self.frame_queue = None
        self.thread = None
        self.stop = None
        self.cur_frame_idx = -1
        self.open()

    def open(self):
        if self.video is None:
            self.video = cv2.VideoCapture(self.video_path)
            if not self.video.isOpened():
                raise Exception('Ensure file is valid video and system dependencies are up to date.\n')
            self.frame_queue = queue.Queue(self.queue_size)
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.decode_thread, args=(self.video, self.frame_queue, self.stop), daemon=True)
            self.thread.start()

    def close(self):
        if self.video is not None:
            self.stop.set()
            self.thread.join()
            self.stop.clear()
            self.video.release()
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.video = None
            self.frame_queue = None
            self.thread = None
            self.stop = None
            self.cur_frame_idx = -1

    def reset(self):
        self.close()
        self.open()

    @property
    def frame_size(self):
        return (
            math.trunc(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            math.trunc(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)

    def read(self):
        frame_idx, frame = self.frame_queue.get()
        if frame_idx is None and frame is None:
            return None
        else:
            self.cur_frame_idx = frame_idx
            return frame

    def decode_thread(self, video, frame_queue, stop):
        frame_idx = 0
        try:
            while not stop.is_set():
                ret, frame = video.read()
                if not ret:
                    break
                if self.dst_size is not None:
                    frame = cv2.resize(frame, self.dst_size, interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame_idx, frame))
                frame_idx += 1
                if self.video_length is not None and frame_idx >= self.video_length:
                    break
        except Exception:
            stop.set()
        finally:
            frame_queue.put((None, None))


class VideoReaderDecord:
    def __init__(self, video_path, video_length=None, dst_size=None, queue_size=4):
        self.video_path = video_path
        self.video_length = video_length
        self.dst_size = dst_size
        self.queue_size = queue_size
        self.video = None
        self.frame_queue = None
        self.thread = None
        self.stop = None
        self.cur_frame_idx = -1
        self.open()

    def open(self):
        if self.video is None:
            self.video = VideoReader(self.video_path)
            if self.video_length is None:
                self.video_length = len(self.video)
            assert self.video_length <= len(self.video)
            self.frame_queue = queue.Queue(self.queue_size)
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.decode_thread, args=(self.video, self.frame_queue, self.stop), daemon=True)
            self.thread.start()

    def close(self):
        if self.video is not None:
            self.stop.set()
            self.thread.join()
            self.stop.clear()
            # self.video.release()
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.video = None
            self.frame_queue = None
            self.thread = None
            self.stop = None
            self.cur_frame_idx = -1

    def reset(self):
        self.close()
        self.open()

    @property
    def fps(self):
        return self.video.get_avg_fps()

    def read(self):
        frame_idx, frame = self.frame_queue.get()
        if frame_idx is None and frame is None:
            return None
        else:
            self.cur_frame_idx = frame_idx
            return frame

    def decode_thread(self, video, frame_queue, stop):
        frame_idx = 0
        try:
            while not stop.is_set():
                frame = video.next()
                frame = frame.numpy() if isinstance(frame, torch.Tensor) else frame.asnumpy()
                if self.dst_size is not None:
                    frame = cv2.resize(frame, self.dst_size, interpolation=cv2.INTER_LINEAR)
                frame_queue.put((frame_idx, frame))
                frame_idx += 1
                if frame_idx >= self.video_length:
                    break
        except Exception:
            stop.set()
        finally:
            frame_queue.put((None, None))

# new avi reader
class VideoReaderAV:
    def __init__(self, video_path, video_length=None, dst_size=None, queue_size=4):
        self.video_path = video_path
        self.video_length = video_length
        self.dst_size = dst_size  # (width, height)
        self.queue_size = queue_size
        self.container = None
        self.frame_queue = None
        self.thread = None
        self.stop = None
        self.cur_frame_idx = -1
        self.open()

    def open(self):
        if self.container is None:
            self.container = av.open(self.video_path)
            self.video_stream = self.container.streams.video[0]
            if self.video_length is None:
                self.video_length = self.video_stream.frames
            self.frame_queue = queue.Queue(self.queue_size)
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.decode_thread, daemon=True)
            self.thread.start()

    def close(self):
        if self.container is not None:
            self.stop.set()
            self.thread.join()
            self.stop.clear()
            self.container.close()
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.container = None
            self.frame_queue = None
            self.thread = None
            self.stop = None
            self.cur_frame_idx = -1

    def reset(self):
        self.close()
        self.open()

    def read(self):
        frame_idx, frame = self.frame_queue.get()
        if frame_idx is None and frame is None:
            return None
        else:
            self.cur_frame_idx = frame_idx
            return frame

    def decode_thread(self):
        frame_idx = 0
        try:
            for packet in self.container.demux(self.video_stream):
                if self.stop.is_set():
                    break
                for frame in packet.decode():
                    if self.stop.is_set():
                        break
                    img = frame.to_ndarray(format='rgb24')
                    if self.dst_size is not None:
                        img = cv2.resize(img, self.dst_size, interpolation=cv2.INTER_LINEAR)
                    self.frame_queue.put((frame_idx, img))
                    frame_idx += 1
                    if self.video_length is not None and frame_idx >= self.video_length:
                        break
                if self.video_length is not None and frame_idx >= self.video_length:
                    break
        except Exception as e:
            self.stop.set()
            print(f"Error in decode_thread: {e}")
        finally:
            self.frame_queue.put((None, None))
