# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,os,pickle,math
import cv2,dlib,time
import numpy as np
from tqdm import tqdm
import json

def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames

# read list of dict string 
def read_datalist(datalist):
    list_of_fid = []
    with open(datalist, "r") as file:
        for line in file:
            dict_from_line = json.loads(line.strip())
            list_of_fid.append(dict_from_line)
    return list_of_fid

def detect_face_landmarks(face_predictor_path, cnn_detector_path, landmark_dir, flist_fn, rank, nshard):
    def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    output_dir = landmark_dir #

    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)
    
    fids = read_datalist(flist_fn)
    num_per_shard = math.ceil(len(fids)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fids = fids[start_id: end_id]
    print(f"{len(fids)} files")
    for fid in tqdm(fids):
        key = fid['key']
        path = fid['wav'].split('.')[0]
        output_fn = os.path.join(output_dir, key+'.pkl')
        video_path = path + '.avi'
        if os.path.exists(output_fn):
            continue
        if not os.path.exists(video_path):
            video_path = video_path = path + '.mp4'
        try:
            frames = load_video(video_path)
        except:
            print(video_path)
            continue
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, 'wb'))
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--landmark', type=str, help='landmark dir')
    parser.add_argument('--datalist', type=str, help='a list of filenames')
    parser.add_argument('--cnn_detector', type=str, help='path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)')
    parser.add_argument('--face_predictor', type=str, help='path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    args = parser.parse_args()
    import skvideo
    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    print(skvideo.getFFmpegPath())
    import skvideo.io
    detect_face_landmarks(args.face_predictor, args.cnn_detector, args.landmark, args.datalist, args.rank, args.nshard)


# python local/detect_landmark.py --landmark data/landmarkdata --datalist data/seg_data/Patient/data.list --cnn_detector detector/mmod_human_face_detector.dat --face_predictor detector/shape_predictor_68_face_landmarks.dat --ffmpeg /mnt/shareEEx/liuxiaokang/miniconda3/envs/AVSR/bin/ffmpeg --rank $i --nshard 20