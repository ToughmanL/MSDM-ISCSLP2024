# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

from typing import Dict, Optional, Tuple
import torch
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.resnet_3d import generate_model

from wenet.utils.common import (IGNORE_ID)


class DYSModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        modarity: 'audiovideo', # audio, video, audiovideo
        depth: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.modarity = modarity
        self.encoder = encoder
        self.resnet3d = generate_model(depth)
        self.crossatt = torch.nn.MultiheadAttention(embed_dim=128, num_heads=2, dropout=0.3)
        self.linear = torch.nn.Linear(128, vocab_size)
        self.dropout = torch.nn.Dropout(0.6)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lengths: torch.Tensor,
        label: torch.Tensor,
    ):
        if self.modarity == 'audio':
            return self.forward_encoder(speech, speech_lengths, label)
        elif self.modarity == 'video':
            return self.forward_3dresnet(video, label)
        else:
            return self.forward_encoder_3dresnet(speech, speech_lengths, video, label)

    def forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        label: torch.Tensor,
    ):
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = encoder_out.mean(dim=1)
        encoder_out = self.dropout(encoder_out)
        encoder_out = self.linear(encoder_out)
        loss = self.loss_fn(encoder_out, label)
        return {"loss": loss}

    def forward_3dresnet(
        self,
        video: torch.Tensor,
        label: torch.Tensor,
    ):
        # 1. 3DResNet
        video = video.permute(0, 2, 1, 3, 4)
        video = self.resnet3d(video)
        video = self.dropout(video)
        video = self.linear(video)
        loss = self.loss_fn(video, label)
        return {"loss": loss}
    
    def forward_encoder_3dresnet(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        video: torch.Tensor,
        label: torch.Tensor,
    ):
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = encoder_out.mean(dim=1)

        # 2. 3DResNet
        video = video.permute(0, 2, 1, 3, 4)
        video = self.resnet3d(video)
        encoder_out = torch.add(encoder_out, video)
        encoder_out = self.dropout(encoder_out)
        encoder_out = self.linear(encoder_out)
        loss = self.loss_fn(encoder_out, label)
        return {"loss": loss}

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) :
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks
        )  # (B, maxlen, encoder_dim)
        encoder_out = encoder_out.mean(dim=1)
        return encoder_out

    def decode(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
        assert speech.shape[0] == speech_lengths.shape[0]
        if self.modarity == 'audio':
            encoder_out = self._forward_encoder(speech, speech_lengths)
        elif self.modarity == 'video':
            video = video.permute(0, 2, 1, 3, 4)
            encoder_out = self.resnet3d(video)
        elif self.modarity == 'audiovideo':
            encoder_out = self._forward_encoder(speech, speech_lengths)
            video = video.permute(0, 2, 1, 3, 4)
            resnet_out = self.resnet3d(video)
            encoder_out = torch.add(encoder_out, resnet_out)
        encoder_out = self.linear(encoder_out)
        results = torch.softmax(encoder_out, dim=1)
        return results

