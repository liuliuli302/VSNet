#!/usr/bin/env python
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo for the evaluation of video summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script takes a random video, selects a random summary
% Then, it evaluates the summary and plots the performance compared to the human summaries
%
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
% date:        05-16-2014
"""
import os
from pathlib import Path
from summe import *
import numpy as np
import random
import json


def raw_json_to_array(json_file):
    # 读取 JSON 文件
    with open(json_file, "r") as f:
        data = json.load(f)

    # 获取最大下标
    max_index = max(int(key) for key in data.keys())

    # 创建一个数组，初始值为 0
    result_array = [0] * (max_index + 1)

    # 填充数组
    for key, value in data.items():
        result_array[int(key)] = value

    return np.array(result_array)


def refine_json_to_array(json_file):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 计算最终数组的长度
    max_outer_index = max(int(key) for key in data.keys())
    max_inner_index = 9  # 假设内层索引是 0 到 9
    result_length = max_outer_index + 10  # 每个外层索引填充16个位置
    
    # 创建结果数组，初始值为 0
    result_array = [0] * result_length

    # 填充数组
    for outer_key, inner_dict in data.items():
        outer_index = int(outer_key)
        for inner_key, value in inner_dict.items():
            inner_index = int(inner_key)
            position = outer_index + inner_index
            result_array[position] = value

    return np.array(result_array)


HOMEDATA = "/mnt/e/dataset/SumMe/SumMe/GT"
HOMEVIDEOS = "/mnt/e/dataset/SumMe/SumMe/videos/"


if __name__ == "__main__":
    # Take a random video and create a random summary for it
    included_extenstions = ["webm"]
    videoList = [
        fn
        for fn in os.listdir(HOMEVIDEOS)
        if any([fn.endswith(ext) for ext in included_extenstions])
    ]
    videoName = videoList[int(round(random.random() * 24))]
    videoName = videoName.split(".")[0]

    # In this example we need to do this to now how long the summary selection needs to be
    gt_file = HOMEDATA + "/" + videoName + ".mat"
    gt_data = scipy.io.loadmat(gt_file)
    nFrames = gt_data.get("nFrames")
    # get the number of frames
    n_frames = nFrames[0][0]

    """Example summary vector"""
    # selected frames set to n (where n is the rank of selection) and the rest to 0
    summary_selections = {}
    summary_selections[0] = np.random.random(size=n_frames) * 20
    summary_selections[0] = list(
        map(
            lambda q: (
                round(q) if (q >= np.percentile(summary_selections[0], 85)) else 0
            ),
            summary_selections[0],
        )
    )

    # ==============================

    
    
    path = Path(
        f"/home/insight/workspace/VSNet/scores/SumMe/refine/{videoName.replace(' ', '_')}.json"
    )
    summary = refine_json_to_array(path)
    summary = summary * 20

    # gt_summary = gt_data.get("gt_score")
    # gt_summary = np.array(gt_summary).squeeze() * 10 + np.random.random(size=n_frames)
    # gt_max = np.max(gt_summary)
    # gt_min = np.min(gt_summary)
    # gt_summary = (gt_summary - gt_min) / (gt_max - gt_min)
    a = np.percentile(summary, 85)
    summary = list(
        map(
            lambda q: (round(q) if (q >= np.percentile(summary, 85)) else 0),
            summary,
        )
    )
    [gt_measure, s_len] = evaluateSummary(summary, videoName, HOMEDATA)
    # ==============================

    """Evaluate"""
    # get f-measure at 15% summary length
    # [f_measure, summary_length] = evaluateSummary(
    #     summary_selections[0], videoName, HOMEDATA
    # )
    # print(f"F-measure :{f_measure} at length {summary_length} at Video {videoName}")

    # """plotting"""
    # methodNames = {"Random"}
    # plotAllResults(summary_selections, methodNames, videoName, HOMEDATA)
