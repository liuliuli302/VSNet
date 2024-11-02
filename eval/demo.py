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
from summe import *
import numpy as np
import random

""" PATHS """

HOMEDATA = os.path.expandvars("/mnt/e/dataset/SumMe/SumMe/GT")
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
    gt_summary = gt_data.get("gt_score")
    gt_summary = np.array(gt_summary).squeeze() * 10 + np.random.random(size=n_frames)
    # gt_max = np.max(gt_summary)
    # gt_min = np.min(gt_summary)
    # gt_summary = (gt_summary - gt_min) / (gt_max - gt_min)
    gt_summary = list(
        map(
            lambda q: (round(q) if (q >= np.percentile(gt_summary, 85)) else 0),
            gt_summary,
        )
    )
    [gt_measure, s_len] = evaluateSummary(
        gt_summary, videoName, HOMEDATA
    )
    # ==============================

    """Evaluate"""
    # get f-measure at 15% summary length
    [f_measure, summary_length] = evaluateSummary(
        summary_selections[0], videoName, HOMEDATA
    )
    print(f"F-measure :{f_measure} at length {summary_length} at Video {videoName}")

    """plotting"""
    methodNames = {"Random"}
    plotAllResults(summary_selections, methodNames, videoName, HOMEDATA)
