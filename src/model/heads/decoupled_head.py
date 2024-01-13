import math
import torch
import torch.nn as nn
from model.layers.network_blocks import BaseConv


class DecoupledHead(nn.Module):
    def __init__(
        self,
        n_anchors=1,
        in_channels=None,
        norm="bn",
        act="silu",
    ):
        super().__init__()
        self.n_anchors = n_anchors

        # age        :  6
        # race       :  3
        # masked     :  2
        # skintone   :  4
        # emotion    :  7
        # gender     :  2
        n_age = self.n_anchors * 6
        n_race = self.n_anchors * 3
        n_masked = self.n_anchors * 2
        n_skintone = self.n_anchors * 4
        n_emotion = self.n_anchors * 7
        n_gender = self.n_anchors * 2

        conv = BaseConv
        self.stems = nn.ModuleList()
        self.age_convs = nn.ModuleList()
        self.age_preds = nn.ModuleList()
        self.race_convs = nn.ModuleList()
        self.race_preds = nn.ModuleList()
        self.masked_convs = nn.ModuleList()
        self.masked_preds = nn.ModuleList()
        self.skintone_convs = nn.ModuleList()
        self.skintone_preds = nn.ModuleList()
        self.emotion_convs = nn.ModuleList()
        self.emotion_preds = nn.ModuleList()
        self.gender_convs = nn.ModuleList()
        self.gender_preds = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        def build_layer(convs, preds, n):
            convs.append(
                nn.Sequential(
                    *[
                        conv(
                            in_channels[0],
                            in_channels[0],
                            ksize=3,
                            stride=1,
                            norm=norm,
                            act=act,
                        ),
                        conv(
                            in_channels[0],
                            in_channels[0],
                            ksize=3,
                            stride=1,
                            norm=norm,
                            act=act,
                        ),
                    ]
                )
            )

            preds.append(
                nn.Conv2d(
                    in_channels[0], n, kernel_size=(1, 1), stride=(1, 1), padding=0
                )
            )

        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels[i], in_channels[0], ksize=1, stride=1, act=act)
            )

            build_layer(self.age_convs, self.age_preds, n_age)
            build_layer(self.race_convs, self.race_preds, n_race)
            build_layer(self.masked_convs, self.masked_preds, n_masked)
            build_layer(self.skintone_convs, self.skintone_preds, n_skintone)
            build_layer(self.emotion_convs, self.emotion_preds, n_emotion)
            build_layer(self.gender_convs, self.gender_preds, n_gender)

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        conv(
                            in_channels[0],
                            in_channels[0],
                            ksize=3,
                            stride=1,
                            norm=norm,
                            act=act,
                        ),
                        conv(
                            in_channels[0],
                            in_channels[0],
                            ksize=3,
                            stride=1,
                            norm=norm,
                            act=act,
                        ),
                    ]
                )
            )

            self.reg_preds.append(
                nn.Conv2d(
                    in_channels[0],
                    self.n_anchors * 4,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )

            self.obj_preds.append(
                nn.Conv2d(
                    in_channels[0],
                    self.n_anchors * 1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )

        self.initialize_biases(self.age_preds, 1e-2)
        self.initialize_biases(self.race_preds, 1e-2)
        self.initialize_biases(self.masked_preds, 1e-2)
        self.initialize_biases(self.skintone_preds, 1e-2)
        self.initialize_biases(self.emotion_preds, 1e-2)
        self.initialize_biases(self.gender_preds, 1e-2)
        self.initialize_biases(self.obj_preds, 1e-2)

    def initialize_biases(self, target, prior_prob):
        for conv in target:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, inputs):
        outputs = []
        for k, (
            age_conv,
            race_conv,
            masked_conv,
            skintone_conv,
            emotion_conv,
            gender_conv,
            reg_conv,
            x,
        ) in enumerate(
            zip(
                self.age_convs,
                self.race_convs,
                self.masked_convs,
                self.skintone_convs,
                self.emotion_convs,
                self.gender_convs,
                self.reg_convs,
                inputs,
            )
        ):
            # Change all inputs to the same channel.
            x = self.stems[k](x)

            age_x = x
            race_x = x
            masked_x = x
            skintone_x = x
            emotion_x = x
            gender_x = x
            reg_x = x

            age_feat = age_conv(age_x)
            race_feat = race_conv(race_x)
            masked_feat = masked_conv(masked_x)
            skintone_feat = skintone_conv(skintone_x)
            emotion_feat = emotion_conv(emotion_x)
            gender_feat = gender_conv(gender_x)

            age_output = self.age_preds[k](age_feat)
            race_output = self.race_preds[k](race_feat)
            masked_output = self.masked_preds[k](masked_feat)
            skintone_output = self.skintone_preds[k](skintone_feat)
            emotion_output = self.emotion_preds[k](emotion_feat)
            gender_output = self.gender_preds[k](gender_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat(
                [
                    reg_output,
                    obj_output,
                    age_output,
                    race_output,
                    masked_output,
                    skintone_output,
                    emotion_output,
                    gender_output,
                ],
                1,
            )
            outputs.append(output)
        return outputs
