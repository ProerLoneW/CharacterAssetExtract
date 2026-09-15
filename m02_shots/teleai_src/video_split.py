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
import functools
import subprocess

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from decord import VideoReader
from tqdm import tqdm
from moviepy.editor import VideoFileClip

from mytools.utils import print_with_timestamp as print
from mytools.utils import sleep_for_test
from video_utils import VideoReaderCV2, VideoReaderDecord, crc64_check, get_video_length, get_video_size, video_reader_check, is_audio_ok, get_video_length  # NOQA
from video_utils import VideoReaderAV
from data_provider import HttpDataProvider, PipeDataProvider
from loguru import logger
sys.path.insert(0, os.path.join(os.getcwd(), "others"))
from scrl.extract_embeddings import run_scrl

def get_max_interval(intervals):
    temp_ = None
    for i in intervals:
        if temp_ is None:
            temp_ = i
        else:
            if (i[1] - i[0]) > (temp_[1] - temp_[0]):
                temp_ = i
    return temp_


class TransNetV2(nn.Module):
    def __init__(
        self,
        F=16,
        L=3,
        S=2,
        D=1024,
        use_many_hot_targets=True,
        use_frame_similarity=True,
        use_color_histograms=True,
        use_mean_pooling=False,
        dropout_rate=0.5,
        use_convex_comb_reg=False,  # not supported
        use_resnet_features=False,  # not supported
        use_resnet_like_top=False,  # not supported
        frame_similarity_on_last_layer=False,
    ):  # not supported
        super(TransNetV2, self).__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplemented("Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0)]
            + [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2**i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(sum([(F * 2**i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, "incorrect input type and/or shape"
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot


class StackedDDCNNV2(nn.Module):
    def __init__(self, in_filters, n_blocks, filters, shortcut=True, use_octave_conv=False, pool_type="avg", stochastic_depth_drop_prob=0.0):  # not supported
        super(StackedDDCNNV2, self).__init__()

        if use_octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, octave_conv=use_octave_conv, activation=functional.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)]
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None, octave_conv=False):  # not supported
        super(DilatedDCNNV2, self).__init__()

        if octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters, filters, dilation_rate, separable=True, octave=False, use_bias=True, kernel_initializer=None):  # not supported  # not supported
        super(Conv3DConfigurable, self).__init__()

        if octave:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")
        if kernel_initializer is not None:
            raise NotImplemented("Kernel initializers are not implemented in Pytorch version of Transnet!")

        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3), dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1), dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(in_filters, filters, kernel_size=3, dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters, similarity_dim=128, lookup_window=101, output_dim=128, stop_gradient=False, use_bias=False):  # not supported
        super(FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplemented("Stop gradient not implemented in Pytorch version of Transnet!")

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat([1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat([batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat([batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat([1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat([batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat([batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities


class TransNetV2Pipeline:
    def __init__(self, model_path):
        state_dict = torch.load(model_path)
        self.model = TransNetV2()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, video_path, video_length=None, video_mode='cv2', dst_size=(48, 27)):
        if video_mode == 'cv2':
            # from giga_datasets import VideoReaderCV2 as VideoReader
            video = VideoReaderCV2(video_path, video_length, dst_size, queue_size=100)
        elif video_mode == 'decord':
            # from giga_datasets import VideoReaderDecord as VideoReader
            video = VideoReaderDecord(video_path, video_length, dst_size, queue_size=100)
        elif video_mode == 'av':
            video = VideoReaderAV(video_path, video_length, dst_size, queue_size=100)
        else:
            assert False
        # video = VideoReader(video_path, video_length, dst_size, queue_size=100)
        predictions = self.predict_video(video)
        if len(predictions) != 0:
            scenes = self.predictions_to_scenes(predictions)
        else:
            scenes = []
        video_length = video.cur_frame_idx + 1
        video.close()
        return scenes, video_length

    def get_frames(self, video, buffers):
        is_end = False
        if len(buffers) == 0:
            for i in range(25):
                frame = video.read()
                if frame is None:
                    is_end = True
                    break
                buffers.append(frame)
            if is_end:
                return None, buffers, is_end
            for i in range(25):
                buffers.insert(0, buffers[0])
        frames = []
        for i in range(50):
            frame = video.read()
            if frame is None:
                is_end = True
                break
            frames.append(frame)
        # assert len(frames) > 0
        frames, buffers = buffers + frames, frames
        return frames, buffers, is_end

    def predict_video(self, video):
        predictions = []
        buffers = []
        is_end = False
        while not is_end:
            frames, buffers, is_end = self.get_frames(video, buffers)
            if frames is None:
                return []
            pred = self.forward_model(frames)
            predictions.append(pred)
        if len(buffers) > 25:
            pred = self.forward_model(buffers)
            predictions.append(pred)
        predictions = np.concatenate(predictions)
        assert len(predictions) == video.cur_frame_idx + 1
        return predictions

    def forward_model(self, frames):
        frame_size = len(frames)
        assert 25 < frame_size <= 100
        for i in range(frame_size, 100):
            frames.append(frames[-1])
        frames = torch.from_numpy(np.stack(frames)).to(self.device)
        with torch.no_grad():
            pred, _ = self.model(frames[None])
        pred = pred[0, 25 : min(frame_size, 75), 0]
        pred = torch.sigmoid(pred).cpu().numpy()
        return pred

    def predictions_to_scenes(self, predictions, threshold=0.5):
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])
        # just fix if all predictions are 1
        if len(scenes) == 0:
            return [0, len(predictions) - 1]
        return scenes

class TransNetV2PipelineV2:
    def __init__(
        self,
        model_path,
        threshold=0.3,
        min_frame_num=75,
        max_frame_num=720,
        enable_two_stage=False,
        reader_queue_size=300,
        enable_soft_cut=False,
    ):
        state_dict = torch.load(model_path)
        self.model = TransNetV2()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = "cpu"
        self.threshold = threshold
        self.min_frame_num = min_frame_num
        self.max_frame_num = max_frame_num
        self.frame_cache = []
        self.enable_two_stage = enable_two_stage
        self.reader_queue_size = reader_queue_size
        self.enable_soft_cut = enable_soft_cut

        if self.enable_soft_cut:
            self.crop_margin = 1  # 新模式：只留1帧做Safety Buffer
            logger.info("TransNetV2: Soft Cut Enabled (Many-hot mode)")
        else:
            self.crop_margin = 3  # 老模式：强制保留原有的3帧裁剪
            logger.info("TransNetV2: Legacy Mode (One-hot mode)")

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(
        self, video_path, video_length=None, video_mode="decord", dst_size=(48, 27)
    ):
        if video_mode == "cv2":
            video = VideoReaderCV2(
                video_path, video_length, dst_size, queue_size=self.reader_queue_size
            )
        elif video_mode == "decord":
            video = VideoReaderDecord(
                video_path, video_length, dst_size, queue_size=self.reader_queue_size
            )
        elif video_mode == "av":
            video = VideoReaderAV(
                video_path, video_length, dst_size, queue_size=self.reader_queue_size
            )
        else:
            assert False

        fps = video.fps
        self.max_frame_num = int(fps) * 20
        # predictions = self.predict_video_stage_1(video)
        # if len(predictions) != 0:
        #     valid_scenes, all_scenes = self.predictions_to_scenes(predictions)
        one_hot_preds, many_hot_preds = self.predict_video_stage_1(video)
        # ################
        # with open("/workspace/res.txt", "w") as f:
        #     for i in range(one_hot_preds.shape[0]):
        #         # print("kkk")
        #         f.write(str(one_hot_preds[i]))
        #         # print("one_hot_preds: ", one_hot_preds[i])
        #         f.write("\n")
        # ################

        if len(one_hot_preds) != 0:
            valid_scenes, all_scenes, cut_type = self.predictions_to_scenes(one_hot_preds, many_hot_preds)
        else:
            valid_scenes = []
            all_scenes = []
            cut_type = []
        video_length = video.cur_frame_idx + 1
        video.close()
        if self.enable_two_stage:
            # 注意：Two-stage 逻辑比较复杂，通常只用于精修。
            # 这里暂时保留原有逻辑，如果开启软切分，建议根据效果决定是否还需要 two-stage
            valid_scenes = self.predict_video_stage_2(valid_scenes, video_length)
            self.frame_cache = []
        return valid_scenes, all_scenes, cut_type, video_length, fps

    def filter_and_split(self, predictions):
        valid_scenes = []
        all_scenes = []       # 记录transnetv2的所有预测结果
        cut_type = []
        for pred in predictions:
            # movie gen video  from meta
            all_scenes.append([pred[0], pred[1]])

            # start_frame = pred[0] + 3
            # end_frame = pred[1] - 3
            # start_frame = pred[0] + self.crop_margin
            # end_frame = pred[1] - self.crop_margin
            start_frame = pred[0]
            end_frame = pred[1]
            clip_lenght = end_frame - start_frame + 1

            # if end_frame > 0 and clip_lenght < self.min_frame_num:
            #     logger.info(
            #         f"ignore clip , length {clip_lenght} < {self.min_frame_num}"
            #     )
            #     continue
            # if start_frame > end_frame or (start_frame == end_frame and end_frame > 0):
            #     logger.info(f"ignore clip , start {start_frame} end  {end_frame}")
            #     continue
            if clip_lenght > self.max_frame_num:
                nr_sub_clip = clip_lenght // self.max_frame_num + 1
                logger.info(
                    f"split clip from {start_frame} to {end_frame} with {nr_sub_clip} clips"
                )
                for n in range(nr_sub_clip):
                    valid_scenes.append(
                        [
                            start_frame + n * self.max_frame_num,
                            min(start_frame + (n + 1) * self.max_frame_num - 1, end_frame),
                        ]
                    )
                    if n != nr_sub_clip - 1:
                        cut_type.append(1)
                    else:
                        cut_type.append(0)
            else:
                valid_scenes.append([start_frame, end_frame])
                cut_type.append(0)
        cut_type[-1] = 2
        return valid_scenes, all_scenes, cut_type

    def get_frames(self, video, buffers):
        is_end = False
        if len(buffers) == 0:
            for i in range(25):
                frame = video.read()
                if frame is None:
                    is_end = True
                    break
                buffers.append(frame)
            if is_end:
                return None, buffers, is_end
            for i in range(25):
                buffers.insert(0, buffers[0])
        frames = []
        for i in range(50):
            frame = video.read()
            if frame is None:
                is_end = True
                break
            frames.append(frame)
        frames, buffers = buffers + frames, frames
        return frames, buffers, is_end

    def predict_video_stage_1(self, video):
        # predictions = []
        one_hot_predictions = []
        many_hot_predictions = []

        buffers = []
        is_end = False
        with tqdm(total=video.video_length, desc="transnetv2") as pbar:
            while not is_end:
                frames, buffers, is_end = self.get_frames(video, buffers)
                if frames is None:
                    # return []
                    return [], []
                # pred = self.forward_model(frames, cache=self.enable_two_stage)
                # predictions.append(pred)
                print("len frame: ", len(frames))
                pred_one, pred_many = self.forward_model(frames, cache=self.enable_two_stage)
                print("pred_one: ", pred_one.shape, pred_many.shape)
                one_hot_predictions.append(pred_one)
                many_hot_predictions.append(pred_many)

                pbar.n = video.cur_frame_idx + 1
                pbar.refresh()
            if len(buffers) > 25:
                # pred = self.forward_model(buffers, cache=self.enable_two_stage)
                # predictions.append(pred)
                pred_one, pred_many = self.forward_model(buffers, cache=self.enable_two_stage)
                one_hot_predictions.append(pred_one)
                many_hot_predictions.append(pred_many)
            pbar.n = video.video_length
            pbar.refresh()
            pbar.close()

        # predictions = np.concatenate(predictions)
        # assert len(predictions) == video.cur_frame_idx + 1
        # return predictions
        
        one_hot_predictions = np.concatenate(one_hot_predictions)
        many_hot_predictions = np.concatenate(many_hot_predictions)
        
        assert len(one_hot_predictions) == video.cur_frame_idx + 1
        print("one_hot_predictions: ", one_hot_predictions)
        return one_hot_predictions, many_hot_predictions


    def predict_video_stage_2(self, scenes, video_length):
        refined_scenes = []
        for scene in scenes:
            start_id, end_id = scene
            start_pad_num = 0
            pad_start_id = start_id - 25
            if pad_start_id < 0:
                start_pad_num = -pad_start_id
                pad_start_id = 0
            pad_end_id = min(video_length - 1, end_id + 25)
            raw_clip = self.frame_cache[pad_start_id : pad_end_id + 1]
            if start_pad_num > 0:
                raw_clip = [raw_clip[0]] * int(start_pad_num) + raw_clip
            raw_clip_length = len(raw_clip)
            sub_input_num = (raw_clip_length - 100) // 50 + 1
            win_start_id = 0
            win_end_id = 100
            preds = []
            for i in range(sub_input_num):
                sub_input = raw_clip[win_start_id:win_end_id]
                win_start_id += 50
                win_end_id = min(raw_clip_length, win_end_id + 50)
                pred = self.forward_model(sub_input, False)
                preds.append(pred)
            preds = np.concatenate(preds)
            sub_scenes, _ = self.predictions_to_scenes(preds)
            if len(sub_scenes) == 0:
                refined_scenes.append(scene)
                continue
            if len(sub_scenes) == 1:
                major_sub_scene = sub_scenes[0]
            else:
                sorted_scenes = sorted(
                    sub_scenes, key=lambda item: item[1] - item[0], reverse=True
                )
                major_sub_scene = sorted_scenes[0]
            refined_scenes.append(
                [start_id + major_sub_scene[0], start_id + major_sub_scene[1]]
            )
        return refined_scenes

    def forward_model(self, frames, cache):
        frame_size = len(frames)
        assert 25 < frame_size <= 100
        for i in range(frame_size, 100):
            frames.append(frames[-1])
        if cache:
            self.frame_cache += frames[25 : min(frame_size, 75)]
        frames = torch.from_numpy(np.stack(frames)).to(self.device)
        # with torch.no_grad():
        #     pred, _ = self.model(frames[None])
        # pred = pred[0, 25 : min(frame_size, 75), 0]
        # pred = torch.sigmoid(pred).cpu().numpy()
        # return pred
        with torch.no_grad():
            # [修改] 捕获所有输出
            one_hot_out, dict_out = self.model(frames[None])
            
        # 提取 One-Hot
        pred_one = one_hot_out[0, 25 : min(frame_size, 75), 0]
        pred_one = torch.sigmoid(pred_one).cpu().numpy()
        
        # [新增] 提取 Many-Hot
        # 你的 TransNetV2 代码中 key 是 "many_hot"
        if isinstance(dict_out, dict) and "many_hot" in dict_out:
            pred_many = dict_out["many_hot"][0, 25 : min(frame_size, 75), 0]
            pred_many = torch.sigmoid(pred_many).cpu().numpy()
        else:
            # Fallback 如果模型没输出 many_hot (例如被关闭了)
            print("Fallback 如果模型没输出 many_hot (例如被关闭了)")
            pred_many = np.zeros_like(pred_one)

        return pred_one, pred_many

    # def predictions_to_scenes(self, predictions):
    #     predictions = (predictions > self.threshold).astype(np.uint8)
    #     scenes = []
    #     t, t_prev, start = -1, 0, 0
    #     for i, t in enumerate(predictions):
    #         if t_prev == 1 and t == 0:
    #             start = i
    #         if t_prev == 0 and t == 1 and i != 0:
    #             scenes.append([start, i])
    #         t_prev = t
    #     if t == 0:
    #         scenes.append([start, i])
    #     if len(scenes) == 0:
    #         return [0, len(predictions) - 1]
    #     valid_scenes, all_scenes = self.filter_and_split(scenes)
    #     return valid_scenes, all_scenes
    def predictions_to_scenes(self, one_hot_preds, many_hot_preds=None):
            # [逻辑分叉]
            
            # 分支 A: 启用软转场且 many_hot 数据存在
            if self.enable_soft_cut and many_hot_preds is not None:
                # =============================================================
                # 核心参数调整
                # =============================================================
                # 1. 激进的阈值：只要概率超过 0.1，就视为潜在的转场/残影
                soft_threshold = 0.05
                
                # 2. 安全膨胀：虽然 0.1 已经很低了，但建议保留 1 帧的“连坐”机制
                #    这意味着如果第 N 帧是 0.11，那么 N-1 和 N+1 也会被切掉
                #    如果你觉得 0.1 已经切得足够多了，可以将此值设为 0
                expansion_width = 1  
                # =============================================================

                is_transition = many_hot_preds > soft_threshold

                # 执行形态学膨胀 (Dilation)
                if expansion_width > 0:
                    kernel = np.ones(2 * expansion_width + 1, dtype=bool)
                    is_transition = np.convolve(is_transition, kernel, mode='same') > 0

                scenes = []
                in_scene = False
                start = 0
                
                for i, is_trans in enumerate(is_transition):
                    if is_trans: # 进入转场区域（脏数据）
                        if in_scene: 
                            end = i - 1
                            if end >= start:
                                scenes.append([start, end])
                            in_scene = False
                    else: # 进入安全区域（干净数据）
                        if not in_scene: 
                            start = i
                            in_scene = True
                
                # 处理结尾
                if in_scene:
                    scenes.append([start, len(is_transition) - 1])
                    
            # 分支 B: 传统模式 (Legacy) - 保持不变
            else:
                predictions = (one_hot_preds > self.threshold).astype(np.uint8)
                scenes = []
                t, t_prev, start = -1, 0, 0
                for i, t in enumerate(predictions):
                    if t_prev == 1 and t == 0:
                        start = i
                    if t_prev == 0 and t == 1 and i != 0:
                        scenes.append([start, i])
                    t_prev = t
                if t == 0:
                    scenes.append([start, i])
            
            # 空列表处理
            if len(scenes) == 0:
                # 如果启用了软切分且结果为空，说明整个视频可能都在震荡或模糊，返回空
                return [], []
            
            valid_scenes, all_scenes, cut_type = self.filter_and_split(scenes)
            return valid_scenes, all_scenes, cut_type

def save_cut(video_path, save_path, start_frame, end_frame):
    print("save cut start ", save_path, start_frame, end_frame)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        if start_frame == 0 and end_frame == 0:
            shutil.copy(video_path, save_path)
            print('{} done'.format(video_path))
        else:
            # _ = VideoReader(video_path)
            ori_fps = VideoFileClip(video_path, audio=False).fps
            start_time = start_frame / ori_fps
            end_time = end_frame / ori_fps
            if is_audio_ok(video_path):
                clip = VideoFileClip(video_path).subclip(start_time, end_time)
                clip.write_videofile(save_path, audio_codec='aac', logger=None)
            else:
                print("audio check failed, save without audio")
                clip = VideoFileClip(video_path, audio=False).subclip(start_time, end_time)
                clip.write_videofile(save_path, logger=None)

            _ = VideoReader(save_path)
            print('{} done'.format(save_path))
    except Exception as e:
        traceback.print_exc()
        print('{} {} {} error'.format(video_path, start_frame, end_frame))
        if os.path.isfile(save_path):
            os.remove(save_path)

    print("save cut end ", save_path, start_frame, end_frame)

func_code = '''
import os
import sys
import shutil
from moviepy.editor import VideoFileClip
from video_utils import is_audio_ok
from mytools.utils import  print_with_timestamp as print
from decord import VideoReader
import traceback
def save_cut(video_path, save_path, start_frame, end_frame):
    print("save cut start ", save_path, start_frame, end_frame)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        if start_frame == 0 and end_frame == 0:
            shutil.copy(video_path, save_path)
            print('{} done'.format(video_path))
        else:
            # _ = VideoReader(video_path)
            ori_fps = VideoFileClip(video_path, audio=False).fps
            start_time = start_frame / ori_fps
            end_time = end_frame / ori_fps
            if is_audio_ok(video_path):
                clip = VideoFileClip(video_path).subclip(start_time, end_time)
                clip.write_videofile(save_path, audio_codec='aac', logger=None)
            else:
                print("audio check failed, save without audio")
                clip = VideoFileClip(video_path, audio=False).subclip(start_time, end_time)
                clip.write_videofile(save_path, logger=None)

            _ = VideoReader(save_path)
            print('{} done'.format(save_path))
    except Exception as e:
        traceback.print_exc()
        print('{} {} {} error'.format(video_path, start_frame, end_frame))

    print("save cut end ", save_path, start_frame, end_frame)
'''
def save_cut_with_timeout(video_path, save_path, start_frame, end_frame, timeout=20*60):
    command = "save_cut('{}', '{}', {}, {})".format(video_path, save_path, start_frame, end_frame)
    # 本地适配（两处，保持逻辑等价）：
    # 1) 上游硬编码 /root/anaconda3/bin/python 在本机不存在 → 改用当前解释器；
    # 2) 子进程不继承本进程 sys.path → 显式传 PYTHONPATH，让 func_code 里的
    #    video_utils / mytools import 解析到本文件所在目录（teleai_src）。
    _here = os.path.dirname(os.path.abspath(__file__))
    _env = dict(os.environ)
    _env["PYTHONPATH"] = os.pathsep.join(
        [_here, os.path.join(_here, "others"), _env.get("PYTHONPATH", "")]
    )
    process = subprocess.Popen([sys.executable, '-c', func_code + "\n" + command], env=_env)
    try:
        return_code = process.wait(timeout=timeout)
        if return_code != 0:
            print(f"进程已结束，返回码: {return_code}")
    except subprocess.TimeoutExpired:
        print("进程执行超时")
        process.kill()
        if os.path.isfile(save_path):
            os.unlink(save_path)

# not work
# def save_cut_with_timeout(video_path, save_path, start_frame, end_frame, timeout=20*60):
#     import concurrent.futures
#     try:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(save_cut, video_path, save_path, start_frame, end_frame)
#             result = future.exception(timeout=timeout)
#             print(result)
#     except concurrent.futures.TimeoutError:
#         print("save_cut timeout...")
#         if os.path.isfile(save_path):
#             os.unlink(save_path)


def main_split_only(pipe: TransNetV2Pipeline, provider, video_path, data, remove_ori_video=False, output_folder=None):
    min_length = int(os.environ.get("SplitMinLength", 75))  # 75
    upload_length = 1
    try:
        video_fps = VideoFileClip(video_path, audio=False).fps
        video_height, video_width = get_video_size(video_path)
    except:
        print(f"Failed to read original video {video_path}, skipping...")
        return None
    time1 = time.time()
    # old version of pipe
    # video_intervals, video_length = pipe(video_path)  # 闭区间

    # Try different video modes for the original video
    video_modes = ['cv2', 'decord', 'av']
    for video_mode in video_modes:
        try:
            video_intervals, video_length = pipe(video_path, video_length=None, video_mode=video_mode)
            # print(f"Successfully processed video {video_path} using video_mode='{video_mode}'")
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Failed to process video {video_path} using video_mode='{video_mode}': {e}")

    if len(video_intervals) == 0 and video_length < 25:  # 小视频直接返回
        return None

    _1st_det_time = time.time() - time1
    video_intervals_temp = []
    for vi in video_intervals:
        clip_l = vi[1] - vi[0] - 1
        if clip_l <= 750:
            video_intervals_temp += [vi]
        else:
            n = clip_l // 750 + 1
            new_l = clip_l // n + (1 if clip_l % n > 0 else 0)
            for _n in range(n):
                video_intervals_temp += [[vi[0] + _n * new_l, min(vi[1], vi[0] + (_n + 1) * new_l)]]
    video_intervals = video_intervals_temp

    if len(video_intervals) <= 1:
        video_intervals = [[0, 0]]
        # frame_intervals = [{}] if len(frame_intervals) == 0 else frame_intervals
        # assert len(video_intervals) == len(frame_intervals)
    count = 0
    files = []
    intervals = []
    clip_frames = []
    video_valid_range = []
    write_video_time = 0
    _2nd_det_time = 0
    # split_info = []
    for ii, clip in enumerate(video_intervals):
        start_frame = clip[0]
        end_frame = clip[1]
        if end_frame > 0:
            start_frame = start_frame + 1
            end_frame = end_frame - 1
        if start_frame > end_frame or (start_frame == end_frame and end_frame > 0):
            continue
        if end_frame > 0 and end_frame - start_frame < min_length:
            continue

        ext = "." + video_path.rsplit(".")[-1]
        # save_path = os.path.join(os.path.dirname(video_path).replace('/data/file/', '/data/clips/'), os.path.basename(video_path).replace(ext, '_{}.mp4'.format(count)))
        # if output_folder is not None:
        #     save_path = os.path.join(output_folder, *save_path.split("/")[-3:])       # 保留三层
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = os.path.join(output_folder, f"{os.path.basename(video_path)[:-4]}_{count}.mp4")
        try:
            time1 = time.time()
            save_cut_with_timeout(video_path, save_path, start_frame, end_frame)
            ctime2 = time.time() - time1
        except:
            traceback.print_exc()
            continue

        if not os.path.isfile(save_path):
            continue

        clip_length = get_video_length(save_path)
        time3 = time.time()

        # old version of pipe
        # scenes_2, video_length_2 = pipe(save_path, clip_length, video_mode='decord')
        
        # Try different video modes for the clipped video
        for video_mode in video_modes:
            try:
                scenes_2, video_length_2 = pipe(save_path, clip_length, video_mode=video_mode)
                # print(f"Successfully processed clip {save_path} using video_mode='{video_mode}'")
                break  # Exit the loop if successful
            except Exception as e:
                 print(f"Failed to process clip {save_path} using video_mode='{video_mode}': {e}")

        ctime4 = time.time() - time3

        if len(scenes_2) <= 1:
            video_valid_range_temp = [0, clip_length]
        else:
            video_valid_range_temp = get_max_interval(scenes_2)
            video_valid_range_temp[1] += 1

        if video_valid_range_temp[1] - video_valid_range_temp[0] <= min_length:
            continue

        video_valid_range += [json.dumps(video_valid_range_temp)]
        count += 1
        clip_frames += [video_length_2]
        files += [save_path]

        if end_frame == 0 and start_frame == 0:
            end_frame = video_length

        intervals += [[start_frame, end_frame]]
        write_video_time += ctime2
        _2nd_det_time += ctime4
        # split_info += [json.dumps(frame_intervals[ii])]

        # save info to data
        if len(files) == upload_length:
            if ('result' not in data) or (not isinstance(data.get('result', None), dict)):
                data['result'] = {}
            data['result']['file'] = files
            data['result']['intervals'] = intervals
            data['result']['fps'] = video_fps  # 帧率
            data['result']['frames'] = video_length  # 帧数
            data['result']['clip_frames'] = clip_frames  # clip 帧数
            data['result']['video_height'] = video_height
            data['result']['video_width'] = video_width
            data['result']['video_length'] = video_length  # 时长
            data['result']['video_valid_range'] = video_valid_range
            data['result']['raw_video_path'] = video_path
            # data['result']['split_info'] = split_info
            # output_pipe.writelines(json.dumps(data) + '\n')
            # output_pipe.flush()
            if provider is not None:
                provider.upload_results(data)
            upload_length = 10
            files = []
            intervals = []
            clip_frames = []
            video_valid_range = []
            # split_info = []
    if len(files) > 0 or upload_length == 1:
        if ('result' not in data) or (not isinstance(data.get('result', None), dict)):
            data['result'] = {}
        data['result']['file'] = files
        data['result']['intervals'] = intervals
        data['result']['fps'] = video_fps  # 帧率
        data['result']['frames'] = video_length  # 帧数
        data['result']['clip_frames'] = clip_frames  # clip 帧数
        data['result']['video_height'] = video_height
        data['result']['video_width'] = video_width
        data['result']['video_length'] = video_length  # 时长
        data['result']['video_valid_range'] = video_valid_range
        data['result']['raw_video_path'] = video_path
        # data['result']['split_info'] = split_info
        # output_pipe.writelines(json.dumps(data) + '\n')
        # output_pipe.flush()
        if provider is not None:
            provider.upload_results(data)

    print('{} total_frames: {}, 1st_det_time: {}, 2nd_det_time:{}, write_clips_time: {}'.format(video_path, video_length, round(_1st_det_time, 2), round(_2nd_det_time, 2), round(write_video_time, 2)))
    if remove_ori_video:  # 删除原视频
        os.remove(video_path)

    return None

def main_split_simple(provider, video_path, data, remove_ori_video=False, output_folder=None):
    min_length = int(os.environ.get("SplitMinLength", 75))  # 75
    max_length = 1800
    upload_length = 1
    try:
        video_fps = VideoFileClip(video_path, audio=False).fps
        video_height, video_width = get_video_size(video_path)
        num_frames = get_video_length(video_path)
    except:
        print(f"Failed to read original video {video_path}, skipping...")
        return None
    num_loops = math.ceil(float(num_frames) / max_length)
    count = 0
    files = []
    intervals = []
    clip_frames = []
    video_valid_range = []
    for i in range(num_loops):
        save_path = os.path.join(output_folder, f"{os.path.basename(video_path)[:-4]}_{count}.mp4")
        if os.path.isfile(save_path):
            print(f"file {save_path} is exists, skip...")
            continue
        count += 1
        start_frame = i * max_length
        end_frame = min(start_frame + max_length - 1, num_frames - 1)
        try:
            save_cut_with_timeout(video_path, save_path, start_frame, end_frame+1)
        except:
            print("[ERROR] -- save_cut_with_timeout")
            traceback.print_exc()
            continue
        if not os.path.isfile(save_path):
            print("Failed to generate ", save_path)
            continue
        try:
            video_length = get_video_length(save_path)
            print("video_length: ", video_length)
        except:
            print("[ERROR] -- Failed to get_video_length")
            continue
        time3 = time.time()

        files += [save_path]
        intervals += [[start_frame, end_frame+1]]
        clip_frames += [end_frame - start_frame + 1]
        video_valid_range += [json.dumps((start_frame, end_frame+1))]


        # save info to data
        if len(files) == upload_length:
            if ('result' not in data) or (not isinstance(data.get('result', None), dict)):
                data['result'] = {}
            data['result']['file'] = files
            data['result']['intervals'] = intervals
            data['result']['fps'] = video_fps  # 帧率
            data['result']['frames'] = video_length  # 帧数
            data['result']['clip_frames'] = clip_frames  # clip 帧数
            data['result']['video_height'] = video_height
            data['result']['video_width'] = video_width
            data['result']['video_length'] = video_length  # 时长
            data['result']['video_valid_range'] = video_valid_range
            data['result']['raw_video_path'] = video_path
            # data['result']['split_info'] = split_info
            # output_pipe.writelines(json.dumps(data) + '\n')
            # output_pipe.flush()
            if provider is not None:
                provider.upload_results(data)
            upload_length = 10
            files = []
            intervals = []
            clip_frames = []
            video_valid_range = []
            # split_info = []
    if len(files) > 0 or upload_length == 10:
        if ('result' not in data) or (not isinstance(data.get('result', None), dict)):
            data['result'] = {}
        data['result']['file'] = files
        data['result']['intervals'] = intervals
        data['result']['fps'] = video_fps  # 帧率
        data['result']['frames'] = video_length  # 帧数
        data['result']['clip_frames'] = clip_frames  # clip 帧数
        data['result']['video_height'] = video_height
        data['result']['video_width'] = video_width
        data['result']['video_length'] = video_length  # 时长
        data['result']['video_valid_range'] = video_valid_range
        data['result']['raw_video_path'] = video_path
        # data['result']['split_info'] = split_info
        # output_pipe.writelines(json.dumps(data) + '\n')
        # output_pipe.flush()
        if provider is not None:
            provider.upload_results(data)

    return None

def main_split(pipe: TransNetV2Pipeline, provider, video_path, data, remove_ori_video=True, output_folder=None):
    min_length = int(os.environ.get("SplitMinLength", 75))  # 75
    upload_length = 1
    try:
        video_fps = VideoFileClip(video_path, audio=False).fps
        video_height, video_width = get_video_size(video_path)
    except:
        print(f"Failed to read original video {video_path}, skipping...")
        return None
    clip_max_length = int(video_fps * 10)
    time1 = time.time()
    # old version of pipe
    # video_intervals, video_length = pipe(video_path)  # 闭区间

    # Try different video modes for the original video
    video_modes = ['cv2', 'decord', 'av']
    for video_mode in video_modes:
        try:
            video_intervals, video_length = pipe(video_path, video_length=None, video_mode=video_mode)
            # print(f"Successfully processed video {video_path} using video_mode='{video_mode}'")
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Failed to process video {video_path} using video_mode='{video_mode}': {e}")

    if len(video_intervals) == 0 and video_length < 25:  # 小视频直接返回
        return None

    _1st_det_time = time.time() - time1
    video_intervals_temp = []
    for vi in video_intervals:
        clip_l = vi[1] - vi[0] - 1
        if clip_l <= clip_max_length:
            video_intervals_temp += [vi]
        else:
            n = clip_l // clip_max_length + 1
            new_l = clip_l // n + (1 if clip_l % n > 0 else 0)
            for _n in range(int(n)):
                video_intervals_temp += [[vi[0] + _n * new_l, min(vi[1], vi[0] + (_n + 1) * new_l)]]
    video_intervals = video_intervals_temp

    if len(video_intervals) <= 1:
        video_intervals = [[0, 0]]
        # frame_intervals = [{}] if len(frame_intervals) == 0 else frame_intervals
        # assert len(video_intervals) == len(frame_intervals)
    count = 0
    files = []
    intervals = []
    clip_frames = []
    video_valid_range = []
    write_video_time = 0
    _2nd_det_time = 0
    # split_info = []
    for ii, clip in enumerate(video_intervals):
        start_frame = clip[0]
        end_frame = clip[1]
        if end_frame > 0:
            start_frame = start_frame + 1
            end_frame = end_frame - 1
        if start_frame > end_frame or (start_frame == end_frame and end_frame > 0):
            continue
        if end_frame > 0 and end_frame - start_frame < min_length:
            continue

        ext = "." + video_path.rsplit(".")[-1]
        save_path = os.path.join(os.path.dirname(video_path).replace('/data/file/', '/data/clips/'), os.path.basename(video_path).replace(ext, '_{}.mp4'.format(count)))
        if output_folder is not None:
            save_path = os.path.join(output_folder, *save_path.split("/")[-3:])       # 保留三层
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            time1 = time.time()
            save_cut_with_timeout(video_path, save_path, start_frame, end_frame)
            ctime2 = time.time() - time1
        except:
            traceback.print_exc()
            continue

        if not os.path.isfile(save_path):
            continue

        clip_length = get_video_length(save_path)
        time3 = time.time()

        # old version of pipe
        # scenes_2, video_length_2 = pipe(save_path, clip_length, video_mode='decord')
        
        # Try different video modes for the clipped video
        for video_mode in video_modes:
            try:
                scenes_2, video_length_2 = pipe(save_path, clip_length, video_mode=video_mode)
                # print(f"Successfully processed clip {save_path} using video_mode='{video_mode}'")
                break  # Exit the loop if successful
            except Exception as e:
                 print(f"Failed to process clip {save_path} using video_mode='{video_mode}': {e}")

        ctime4 = time.time() - time3

        if len(scenes_2) <= 1:
            video_valid_range_temp = [0, clip_length]
        else:
            video_valid_range_temp = get_max_interval(scenes_2)
            video_valid_range_temp[1] += 1

        if video_valid_range_temp[1] - video_valid_range_temp[0] <= min_length:
            continue

        video_valid_range += [json.dumps(video_valid_range_temp)]
        count += 1
        clip_frames += [video_length_2]
        files += [save_path.replace('/data/clips/', '')]

        if end_frame == 0 and start_frame == 0:
            end_frame = video_length

        intervals += [[start_frame, end_frame]]
        write_video_time += ctime2
        _2nd_det_time += ctime4
        # split_info += [json.dumps(frame_intervals[ii])]

        # save info to data
        if len(files) == upload_length:
            if ('result' not in data) or (not isinstance(data.get('result', None), dict)):
                data['result'] = {}
            data['result']['file'] = files
            data['result']['intervals'] = intervals
            data['result']['fps'] = video_fps  # 帧率
            data['result']['frames'] = video_length  # 帧数
            data['result']['clip_frames'] = clip_frames  # clip 帧数
            data['result']['video_height'] = video_height
            data['result']['video_width'] = video_width
            data['result']['video_length'] = video_length  # 时长
            data['result']['video_valid_range'] = video_valid_range
            data['result']['raw_video_path'] = video_path
            # data['result']['split_info'] = split_info
            # output_pipe.writelines(json.dumps(data) + '\n')
            # output_pipe.flush()
            if provider is not None:
                provider.upload_results(data)
            upload_length = 10
            files = []
            intervals = []
            clip_frames = []
            video_valid_range = []
            # split_info = []
    if len(files) > 0 or upload_length == 1:
        if ('result' not in data) or (not isinstance(data.get('result', None), dict)):
            data['result'] = {}
        data['result']['file'] = files
        data['result']['intervals'] = intervals
        data['result']['fps'] = video_fps  # 帧率
        data['result']['frames'] = video_length  # 帧数
        data['result']['clip_frames'] = clip_frames  # clip 帧数
        data['result']['video_height'] = video_height
        data['result']['video_width'] = video_width
        data['result']['video_length'] = video_length  # 时长
        data['result']['video_valid_range'] = video_valid_range
        data['result']['raw_video_path'] = video_path
        # data['result']['split_info'] = split_info
        # output_pipe.writelines(json.dumps(data) + '\n')
        # output_pipe.flush()
        if provider is not None:
            provider.upload_results(data)

    print('{} total_frames: {}, 1st_det_time: {}, 2nd_det_time:{}, write_clips_time: {}'.format(video_path, video_length, round(_1st_det_time, 2), round(_2nd_det_time, 2), round(write_video_time, 2)))
    if remove_ori_video:  # 删除原视频
        os.remove(video_path)

    return None

def has_audio(vid):
    cmd=["/usr/bin/ffprobe","-print_format","json","-show_streams",vid]
    info=json.loads(subprocess.check_output(cmd))
    return any(s["codec_type"]=="audio" for s in info["streams"])


FFMPEG_BIN = "/usr/bin/ffmpeg"
HD_CUT_BATCH_MAX_CLIPS = 32
HD_CUT_ENCODER_THREADS = 32
UHD_CUT_BATCH_MAX_CLIPS = 16
UHD_CUT_ENCODER_THREADS = 16
DEFAULT_CUT_BATCH_MAX_CLIPS = HD_CUT_BATCH_MAX_CLIPS
DEFAULT_CUT_ENCODER_THREADS = HD_CUT_ENCODER_THREADS
DEFAULT_CUT_BATCH_MAX_SECONDS = 600.0
DEFAULT_CUT_SEEK_MARGIN_FRAMES = 50


def _get_cut_resources(video_path):
    try:
        video_height, video_width = get_video_size(video_path)
    except Exception as exc:
        logger.warning(f"failed to get video resolution for {video_path}: {exc}; use UHD cut settings")
        return UHD_CUT_BATCH_MAX_CLIPS, UHD_CUT_ENCODER_THREADS

    is_uhd = video_width >= 3840 or video_height >= 2160
    if is_uhd:
        batch_max_clips = UHD_CUT_BATCH_MAX_CLIPS
        encoder_threads = UHD_CUT_ENCODER_THREADS
    else:
        batch_max_clips = HD_CUT_BATCH_MAX_CLIPS
        encoder_threads = HD_CUT_ENCODER_THREADS

    logger.info(
        f"cut resources for {video_width}x{video_height}: "
        f"batch_max_clips={batch_max_clips}, encoder_threads={encoder_threads}"
    )
    return batch_max_clips, encoder_threads


def _iter_cut_specs(scene_boundaries, cut_type, video_name, frame_rate, if_second_cut, add_frame):
    specs = []
    for i, (start_frame, end_frame) in enumerate(scene_boundaries):
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame - start_frame <= 10:
            continue
        ori_frame_range = [start_frame, end_frame]
        if if_second_cut:
            ori_frame_range[0] += add_frame
            ori_frame_range[1] += add_frame
        specs.append({
            "index": i,
            "name": video_name + f"_{i:05d}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "cut_type": cut_type[i] if i < len(cut_type) else None,
            "ori_frame_range": ori_frame_range,
        })
    return specs


def _iter_cut_batches(specs, frame_rate, batch_max_clips, batch_max_seconds):
    batch = []
    for spec in specs:
        if not batch:
            batch = [spec]
            continue
        window_seconds = (spec["end_frame"] - batch[0]["start_frame"] + 1) / frame_rate
        if len(batch) >= batch_max_clips or window_seconds > batch_max_seconds:
            yield batch
            batch = [spec]
        else:
            batch.append(spec)
    if batch:
        yield batch


def _build_batch_cut_filter(batch, batch_seek_frame, frame_rate, audio_enabled):
    filter_parts = []
    stream_count = len(batch)
    if stream_count == 1:
        spec = batch[0]
        rel_start = spec["start_frame"] - batch_seek_frame
        rel_end = spec["end_frame"] + 1 - batch_seek_frame
        filter_parts.append(
            f"[0:v]trim=start_frame={rel_start}:end_frame={rel_end},setpts=PTS-STARTPTS[v0]"
        )
        if audio_enabled:
            filter_parts.append(
                f"[0:a]atrim=start={rel_start / frame_rate:.6f}:end={rel_end / frame_rate:.6f},asetpts=PTS-STARTPTS[a0]"
            )
        return ";".join(filter_parts)

    video_outputs = "".join(f"[v{i}src]" for i in range(stream_count))
    filter_parts.append(f"[0:v]split={stream_count}{video_outputs}")
    if audio_enabled:
        audio_outputs = "".join(f"[a{i}src]" for i in range(stream_count))
        filter_parts.append(f"[0:a]asplit={stream_count}{audio_outputs}")

    for i, spec in enumerate(batch):
        rel_start = spec["start_frame"] - batch_seek_frame
        rel_end = spec["end_frame"] + 1 - batch_seek_frame
        filter_parts.append(
            f"[v{i}src]trim=start_frame={rel_start}:end_frame={rel_end},setpts=PTS-STARTPTS[v{i}]"
        )
        if audio_enabled:
            filter_parts.append(
                f"[a{i}src]atrim=start={rel_start / frame_rate:.6f}:end={rel_end / frame_rate:.6f},asetpts=PTS-STARTPTS[a{i}]"
            )
    return ";".join(filter_parts)


def _write_cut_label(spec, output_label_folder, frame_rate):
    data_dict = {
        "name": spec["name"],
        "ori_frame_range": spec["ori_frame_range"],
        "cut_type": spec["cut_type"],
        "ori_fps": frame_rate,
    }
    output_label_name = os.path.join(output_label_folder, spec["name"] + ".json")
    with open(output_label_name, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)
    return output_label_name


def _run_single_trim_cut(video_path, spec, output_video_filename, frame_rate, audio_enabled, encoder_threads):
    cmd = [
        FFMPEG_BIN,
        '-hide_banner',
        '-loglevel', 'quiet',
        "-i", video_path,
        "-filter:v", f"trim=start_frame={spec['start_frame']}:end_frame={spec['end_frame'] + 1},setpts=PTS-STARTPTS",
        "-c:v", "libx264",
        "-threads:v", str(encoder_threads),
    ]
    if audio_enabled:
        cmd.extend([
            "-filter:a",
            f"atrim=start={spec['start_frame'] / frame_rate:.6f}:end={(spec['end_frame'] + 1) / frame_rate:.6f},asetpts=PTS-STARTPTS",
            "-c:a", "aac",
        ])
    cmd += ["-y", output_video_filename]
    return subprocess.run(cmd)


def save_video_ffmpeg(
    scene_boundaries,
    cut_type,
    video_path,
    fps,
    output_video_folder,
    output_label_folder,
    if_second_cut,
    add_frame,
    batch_max_clips=DEFAULT_CUT_BATCH_MAX_CLIPS,
    batch_max_seconds=DEFAULT_CUT_BATCH_MAX_SECONDS,
    seek_margin_frames=DEFAULT_CUT_SEEK_MARGIN_FRAMES,
    encoder_threads=DEFAULT_CUT_ENCODER_THREADS,
):
    frame_rate = float(fps) if fps else 25.0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_label_file_list = []
    output_video_list = []
    os.makedirs(output_video_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    specs = _iter_cut_specs(scene_boundaries, cut_type, video_name, frame_rate, if_second_cut, add_frame)
    if not specs:
        print("视频切割完成！")
        return output_label_file_list, output_video_list

    audio_enabled = has_audio(video_path)
    ffmpeg_call_count = 0
    fallback_count = 0
    write_video_time = 0.0

    batches = list(_iter_cut_batches(specs, frame_rate, batch_max_clips, batch_max_seconds))
    for batch_index, batch in enumerate(batches, start=1):
        logger.info(
            f"start cut batch {batch_index}/{len(batches)}: "
            f"clips={len(batch)}, range={batch[0]['index']}-{batch[-1]['index']}"
        )
        batch_seek_frame = max(0, batch[0]["start_frame"] - int(seek_margin_frames))
        batch_seek_time = batch_seek_frame / frame_rate
        ffmpeg_command = [
            FFMPEG_BIN,
            '-hide_banner',
            '-loglevel', 'quiet',
            "-ss", f"{batch_seek_time:.6f}",
            "-accurate_seek",
            "-i", video_path,
            "-filter_complex", _build_batch_cut_filter(batch, batch_seek_frame, frame_rate, audio_enabled),
        ]

        for i, spec in enumerate(batch):
            output_video_filename = os.path.join(output_video_folder, spec["name"] + ".mp4")
            ffmpeg_command.extend(["-map", f"[v{i}]"])
            if audio_enabled:
                ffmpeg_command.extend(["-map", f"[a{i}]"])
            ffmpeg_command.extend([
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-threads:v", str(encoder_threads),
            ])
            if audio_enabled:
                ffmpeg_command.extend(["-c:a", "aac"])
            ffmpeg_command.extend(["-y", output_video_filename])

        time1 = time.time()
        proc = subprocess.run(ffmpeg_command)
        write_video_time += time.time() - time1
        ffmpeg_call_count += 1
        logger.info(
            f"finish cut batch {batch_index}/{len(batches)} in "
            f"{time.time() - time1:.2f}s, returncode={proc.returncode}"
        )

        if proc.returncode != 0:
            logger.warning(f"batch ffmpeg cut failed, fallback to single trim for {len(batch)} clips")
            for spec in batch:
                output_video_filename = os.path.join(output_video_folder, spec["name"] + ".mp4")
                if os.path.exists(output_video_filename):
                    os.remove(output_video_filename)
                time1 = time.time()
                fallback_proc = _run_single_trim_cut(video_path, spec, output_video_filename, frame_rate, audio_enabled, encoder_threads)
                write_video_time += time.time() - time1
                ffmpeg_call_count += 1
                fallback_count += 1
                if fallback_proc.returncode != 0:
                    logger.warning(f"single trim fallback failed: {output_video_filename}")

        for spec in batch:
            output_video_filename = os.path.join(output_video_folder, spec["name"] + ".mp4")
            if not os.path.isfile(output_video_filename):
                logger.warning(f"skip missing cut output: {output_video_filename}")
                continue
            output_video_list.append(output_video_filename)
            output_label_file_list.append(_write_cut_label(spec, output_label_folder, frame_rate))

    logger.info(
        f"batch cut finished: clips={len(output_video_list)}, "
        f"ffmpeg_calls={ffmpeg_call_count}, fallback_count={fallback_count}, "
        f"write_video_time={write_video_time:.2f}s"
    )
    print("视频切割完成！")
    return output_label_file_list, output_video_list


def main_split_v2(pipe: TransNetV2PipelineV2, scrl_encoder, scrl_BiLSTM, config:dict, video_path, data, remove_ori_video=True):
    output_video_folder = os.path.join(config["target_path"], config['version'], "video", "data")
    output_label_folder = os.path.join(config["target_path"], config['version'], config["label_dir"])
    os.makedirs(output_video_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    video_tmp_25fps_folder = os.path.join(config["target_path"], config['version'], "video", "data_25fps")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_tmp_25fps_filename = os.path.join(video_tmp_25fps_folder, video_name + ".mp4")

    os.makedirs(video_tmp_25fps_folder, exist_ok=True)
    ffmpeg_command = [
        FFMPEG_BIN,
        '-hide_banner',
        '-loglevel', 'quiet',
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", "fps=25,setpts=N/(25*TB)",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "1",
        "-y",
        video_tmp_25fps_filename,
    ]
    normalize_start = time.time()
    normalize_proc = subprocess.run(ffmpeg_command)
    logger.info(f"normalize to 25fps finished in {time.time() - normalize_start:.2f}s: {video_tmp_25fps_filename}")
    if normalize_proc.returncode != 0:
        raise RuntimeError(f"normalize to 25fps failed: {video_path}")

    logger.info(f"start cut scene for {video_tmp_25fps_filename}")
    video_modes = ["decord", "cv2", "av"]
    for video_mode in video_modes:
        try:
            valid_scenes, all_scenes, cut_type, video_length, fps = pipe(video_tmp_25fps_filename, video_mode=video_mode)
            break
        except Exception as e:
            logger.warning(f"can not parse video {video_tmp_25fps_filename}: {e}")
            valid_scenes = []
            all_scenes = []
            cut_type = []
            video_length = 0
            fps = 0
    logger.info(f"finish cut scene for {video_tmp_25fps_filename} with {len(valid_scenes)} valid clips and {len(all_scenes)} all clips")
    batch_max_clips, encoder_threads = _get_cut_resources(video_tmp_25fps_filename)
    output_label_file_list, output_video_list = save_video_ffmpeg(
        valid_scenes,
        cut_type,
        video_tmp_25fps_filename,
        fps,
        output_video_folder,
        output_label_folder,
        False,
        0,
        batch_max_clips=batch_max_clips,
        batch_max_seconds=float(config.get("cut_batch_max_seconds", DEFAULT_CUT_BATCH_MAX_SECONDS)),
        seek_margin_frames=int(config.get("cut_seek_margin_frames", DEFAULT_CUT_SEEK_MARGIN_FRAMES)),
        encoder_threads=encoder_threads,
    )
    # if os.path.exists(video_tmp_25fps_filename):
    #     os.remove(video_tmp_25fps_filename)

    if not output_video_list:
        logger.warning(f"no cut output for {video_path}")
        return None

    # double_split_output_video_list = []
    # double_split_output_label_file_list = []
    # # 遍历原始视频列表（复制一份列表，避免迭代中修改原列表导致问题）
    # original_video_list = output_video_list.copy()

    # for output_video_path,output_label_file_last in zip(original_video_list,output_label_file_list):
    #     # print("output_video_list len: ", len(original_video_list))
    #     valid_scenes, all_scenes, cut_type, video_length, fps = pipe(output_video_path, video_mode="decord")
    #     # print("output_video_path: ", output_video_path)
    #     # print("valid_scenes: ", valid_scenes)

    #     if len(valid_scenes) > 1:
    #         with open(output_label_file_last, "r", encoding="utf-8") as f:
    #             last_label_data = json.load(f)
    #         add_frame = 0
    #         if "ori_frame_range" in last_label_data:
    #             add_frame = last_label_data["ori_frame_range"][0]
    #         # 切割视频，得到新的片段和标签文件
    #         double_split_output_label_files, double_split_output_videos = \
    #             save_video_ffmpeg(valid_scenes, cut_type, output_video_path, fps, output_video_folder, output_label_folder, True, add_frame)

    #         # 把新生成的片段加入最终结果列表
    #         if len(double_split_output_videos) != 0:
    #             double_split_output_video_list.extend(double_split_output_videos)
    #             double_split_output_label_file_list.extend(double_split_output_label_files)

    #             # 【关键修改】只删除当前处理的原视频文件，不删整个列表！
    #             if os.path.exists(output_video_path):
    #                 os.remove(output_video_path)
    #             if os.path.exists(output_label_file_last):
    #                 os.remove(output_label_file_last)

    # # 循环结束后，更新列表为切割后的新视频列表
    # output_video_list = double_split_output_video_list
    # output_label_file_list = double_split_output_label_file_list
    
    # 镜头聚合
    logger.info(f"start run_scrl for {len(output_video_list)} clips")
    scrl_result = run_scrl(scrl_encoder, scrl_BiLSTM, output_video_list, output_label_folder)
    logger.info("finish run_scrl")
    boundary_indices = scrl_result['data']['boundary_indices']

    for boundary_id in boundary_indices:
        if boundary_id > len(output_label_file_list) - 1:
            break
        output_lable_file_path = output_label_file_list[boundary_id]
        with open(output_lable_file_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        with open(output_lable_file_path, "w", encoding="utf-8") as f:
            if data_dict["cut_type"] != 1:
                data_dict["cut_type"] = 2
            json.dump(data_dict, f, indent=4)
        
            

def main():
    print('视频转场检测v0.0.49.gpu use http')
    sleep_for_test()

    pipe = TransNetV2Pipeline('/home/model-server/dev/data_platform/models/transnetv2/transnetv2-pytorch-weights.pth').to('cuda:0')

    # provider = PipeDataProvider()
    local_port = os.environ.get("LOCAL_PORT", 8080)
    provider = HttpDataProvider(host="http://localhost:{}".format(local_port))       # test 9991
    provider.prepare()
    operator_func = functools.partial(main_split, pipe, provider, remove_ori_video=True)
    provider.process_data(operator_func)
    print("all done")



def test_local():
    pipe = TransNetV2Pipeline('/home/model-server/dev/data_platform/models/transnetv2/transnetv2-pytorch-weights.pth').to('cuda:0')

    # data = {}
    # video_path = "/shared_disk/users/liuzhichao/data/file/VideoSpider/AutoUploadStandard/spider_0927/历史频道_纪录片_6_BV1wx41157bD_10_0927.mp4"
    # crc64_check(video_path)
    # video_reader_check(video_path)
    # main_split(pipe, None, video_path, data, remove_ori_video=False)
    # print(data)

    video_path = "/data/file/VideoSpider/AutoUploadStandard/spider_0927/历史频道_纪录片_6_BV1wx41157bD_10_0927.mp4"
    save_path = "/data/file/VideoSpider/AutoUploadStandard/spider_0927/历史频道_纪录片_6_BV1wx41157bD_10_0927_161.mp4"
    save_cut_with_timeout(video_path, save_path, 39768, 40061, timeout=30)

if __name__ == "__main__":
    # test_local()
    main()

