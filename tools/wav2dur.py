#!/usr/bin/env python3
# encoding: utf-8

import sys

import torchaudio
torchaudio.set_audio_backend("sox_io")

scp = sys.argv[1]
dur_scp = sys.argv[2]

if 'scp' in scp:
    with open(scp, 'r') as f, open(dur_scp, 'w') as fout:
        cnt = 0
        total_duration = 0
        for l in f:
            items = l.strip().split()
            wav_id = items[0]
            fname = items[1]
            cnt += 1
            waveform, rate = torchaudio.load(fname)
            frames = len(waveform[0])
            duration = frames / float(rate)
            total_duration += duration
            fout.write('{} {}\n'.format(wav_id, duration))
        fout.write('alldata {}\n'.format(total_duration))
        print('process {} utts'.format(cnt))
        print('total {} s'.format(total_duration))
elif 'segments' in scp:
    with open(scp, 'r') as f, open(dur_scp, 'w') as fout:
        cnt = 0
        total_duration = 0
        for l in f:
            items = l.strip().split()
            wav_id = items[0]
            s_start = float(items[2])
            s_end = float(items[3])
            cnt += 1
            duration = s_end - s_start
            total_duration += duration
            fout.write('{} {}\n'.format(wav_id, duration))
        fout.write('alldata {}\n'.format(total_duration))
        print('process {} utts'.format(cnt))
        print('total {} s'.format(total_duration))

