import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.losses.iou_loss import bboxes_iou
from utils.bbox import xywh2xyxy


class YOLOv7Loss(nn.Module):
    def __init__(
            self,
            strides,
            anchors,
            obj_ratio,
            box_ratio,
            age_ratio,
            race_ratio,
            gender_ratio,
            masked_ratio,
            skintone_ratio,
            emotion_ratio,
            label_smoothing=0,
            focal_g=0.0,
        ):

        super(YOLOv7Loss, self).__init__()

        self.anchors = torch.tensor(anchors)
        self.strides = strides

        self.nl = len(strides)
        self.na = len(anchors[0])

        # score      :  1
        # box        :  4
        # age        :  6
        # race       :  3
        # masked     :  2
        # skintone   :  4
        # emotion    :  7
        # gender     :  2
        self.ch = 1 + 4 + 6 + 3 + 2 + 4 + 7 + 2

        self.balance = [0.4, 1.0, 4]
        
        self.box_ratio = box_ratio
        self.obj_ratio = obj_ratio
        self.age_ratio = age_ratio
        self.race_ratio = race_ratio
        self.gender_ratio = gender_ratio
        self.masked_ratio = masked_ratio
        self.skintone_ratio = skintone_ratio
        self.emotion_ratio = emotion_ratio

        self.threshold = 4.0

        self.grids = [torch.zeros(1)] * len(strides)
        self.anchor_grid = self.anchors.clone().view(self.nl, 1, -1, 1, 1, 2)

        self.cp, self.cn = smooth_BCE(eps=label_smoothing)

        self.gr = 1

    def __call__(self, inputs, targets):
        # shape of inputs: [batch, ch * anchor, h, w]

        batch_size = inputs[0].shape[0]

        # [batch, ch * anchor, h, w] -> [batch, anchor, h, w, ch]
        for i in range(self.nl):
            prediction = inputs[i].view(
                inputs[i].size(0), self.na, self.ch, inputs[i].size(2), inputs[i].size(3)
            ).permute(0, 1, 3, 4, 2).contiguous()
            inputs[i] = prediction

        # inference
        if not self.training:
            preds = []
            for i in range(self.nl):
                pred = inputs[i].sigmoid()
                h, w = pred.shape[2:4]
                # Three steps to localize predictions: grid, shifts of x and y, grid with stride
                if self.grids[i].shape[2:4] != pred.shape[2:4]:
                    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
                    grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                    self.grids[i] = grid
                else:
                    grid = self.grids[i]

                pred[..., :2] = (pred[..., :2] * 2. - 0.5 + grid) * self.strides[i]
                pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * self.anchor_grid[i].type_as(pred)
                pred = pred.reshape(batch_size, -1, self.ch)
                preds.append(pred)

            # preds: [batch_size, all predictions, n_ch]
            predictions = torch.cat(preds, 1)
            # from (cx,cy,w,h) to (x,y,w,h)
            # box_corner = predictions.new(predictions.shape)
            # box_corner = box_corner[:, :, 0:4]
            # box_corner[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
            # box_corner[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
            # box_corner[:, :, 2] = predictions[:, :, 2]
            # box_corner[:, :, 3] = predictions[:, :, 3]
            # predictions[:, :, :4] = box_corner[:, :, :4]
            return predictions

        # Compute loss
        # Processing ground truth to tensor (img_idx, class, cx, cy, w, h)

        gts_list = []
        for img_idx in range(batch_size):
            target = targets[img_idx]

            gt_boxes = target['boxes']
            gt_age      = target['age'].unsqueeze(-1)
            gt_race     = target['race'].unsqueeze(-1)
            gt_masked   = target['masked'].unsqueeze(-1)
            gt_skintone = target['skintone'].unsqueeze(-1)
            gt_emotion  = target['emotion'].unsqueeze(-1)
            gt_gender   = target['gender'].unsqueeze(-1)

            gt_img_ids = torch.ones_like(gt_age).type_as(gt_age) * img_idx
            
            # [nboxes, img_idx + boxes + age + race + masked + skintone + emotion + gender]
            gt = torch.cat((gt_img_ids, gt_boxes, gt_age, gt_race, gt_masked, gt_skintone,
                            gt_emotion, gt_gender), dim=1)

            gts_list.append(gt)

        targets = torch.cat(gts_list, 0)

        bs, as_, gjs, gis, targets, anchors = self.build_targets(inputs, targets)

        age_loss = torch.zeros(1).type_as(inputs[0])
        race_loss = torch.zeros(1).type_as(inputs[0])
        masked_loss = torch.zeros(1).type_as(inputs[0])
        skintone_loss = torch.zeros(1).type_as(inputs[0])
        emotion_loss = torch.zeros(1).type_as(inputs[0])
        gender_loss = torch.zeros(1).type_as(inputs[0])
        
        box_loss = torch.zeros(1).type_as(inputs[0])
        obj_loss = torch.zeros(1).type_as(inputs[0])

        for i, prediction in enumerate(inputs):
            # image, anchor, gridy, gridx
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
            tobj = torch.zeros_like(prediction[..., 0]).type_as(prediction)  # target obj

            n = b.shape[0]
            if n:
                prediction_pos = prediction[b, a, gj, gi]  # prediction subset corresponding to targets

                grid = torch.stack([gi, gj], dim=1)

                # decode, get prediction results
                xy = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                wh = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                box = torch.cat((xy, wh), 1)

                # process the real box and map it to the feature layer
                selected_tbox = targets[i][:, 1:5] / self.strides[i]
                selected_tbox[:, :2] = selected_tbox[:, :2] - grid.type_as(prediction)

                # calculate the regression loss of the predicted box and the real box
                iou = bbox_iou(box.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                box_loss += (1.0 - iou).mean()
                # -------------------------------------------#
                # get the gt of the confidence loss according to the iou of the prediction result
                # -------------------------------------------#
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # -------------------------------------------#
                # calculate the classification loss of the matched positive sample
                # -------------------------------------------#
                selected_age      = targets[i][:, 5].long() # 6
                selected_race     = targets[i][:, 6].long() # 3
                selected_masked   = targets[i][:, 7].long() # 1
                selected_skintone = targets[i][:, 8].long() # 4
                selected_emotion  = targets[i][:, 9].long() # 7
                selected_gender   = targets[i][:, 10].long() # 1

                # age loss
                t = torch.full_like(prediction_pos[:, 5:11], self.cn).type_as(prediction)  # targets
                t[range(n), selected_age] = self.cp
                # age_weight = torch.tensor([0.011, 0.0771, 0.1295, 0.1939, 0.2305, 0.358]).type_as(prediction)
                age_loss += F.cross_entropy(prediction_pos[:, 5:11], t)  # BCE

                # race loss
                t = torch.full_like(prediction_pos[:, 11:14], self.cn).type_as(prediction)  # targets
                t[range(n), selected_race] = self.cp
                # race_weight = torch.tensor([0.08, 0.0843, 0.8357]).type_as(prediction)
                race_loss += F.cross_entropy(prediction_pos[:, 11:14], t)  # BCE
                
                # masked loss
                t = torch.full_like(prediction_pos[:, 14:16], self.cn).type_as(prediction)  # targets
                t[range(n), selected_masked] = self.cp
                # masked_weight = torch.tensor([0.0329, 0.9671]).type_as(prediction)
                masked_loss += F.cross_entropy(prediction_pos[:, 14:16], t)  # BCE
                
                # skintone loss
                t = torch.full_like(prediction_pos[:, 16:20], self.cn).type_as(prediction)  # targets
                t[range(n), selected_skintone] = self.cp
                # skintone_weight = torch.tensor([0.0209, 0.0593, 0.2742, 0.6456]).type_as(prediction)
                skintone_loss += F.cross_entropy(prediction_pos[:, 16:20], t)  # BCE

                # emotion loss
                t = torch.full_like(prediction_pos[:, 20:27], self.cn).type_as(prediction)
                t[range(n), selected_emotion] = self.cp
                # emotion_weight = torch.tensor([0.0042, 0.008, 0.1023, 0.1218, 0.1283, 0.2944, 0.3409]).type_as(prediction)
                emotion_loss += F.cross_entropy(prediction_pos[:, 20:27], t)  # BCE

                # gender loss
                t = torch.full_like(prediction_pos[:, 27:29], self.cn).type_as(prediction)
                t[range(n), selected_gender] = self.cp
                # gender_weight = torch.tensor([0.6127, 0.3127]).type_as(prediction)
                gender_loss += F.cross_entropy(prediction_pos[:, 27:29], t)  # BCE

            # -------------------------------------------#
            # calculate the confidence loss of whether the target exists
            # and multiply by the ratio of each feature layer
            # -------------------------------------------#
            obj_loss += F.binary_cross_entropy_with_logits(prediction[..., 4], tobj) * self.balance[i]  # obj loss
        # -------------------------------------------#
        # multiply the loss of each part by the ratio
        # after adding them all up, multiply by batch_size
        # -------------------------------------------#
        scaled_box_loss = box_loss * self.box_ratio
        scaled_obj_loss = obj_loss * self.obj_ratio
        scaled_age_loss = age_loss * self.age_ratio
        scaled_race_loss = race_loss * self.race_ratio
        scaled_masked_loss = masked_loss * self.masked_ratio
        scaled_skintone_loss = skintone_loss * self.skintone_ratio
        scaled_emotion_loss = emotion_loss * self.emotion_ratio
        scaled_gender_loss = gender_loss * self.gender_ratio

        loss = scaled_box_loss + scaled_obj_loss + scaled_age_loss \
            + scaled_race_loss + scaled_masked_loss + scaled_skintone_loss \
            + scaled_emotion_loss + scaled_gender_loss

        losses = {
            'loss': loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'age_loss': age_loss,
            'race_loss': race_loss,
            'masked_loss': masked_loss,
            'skintone_loss': skintone_loss,
            'emotion_loss': emotion_loss,
            'gender_loss': gender_loss,

        }

        return losses

    def build_targets(self, predictions, targets):

        # indice: [img_idx, anchor_idx, grid_x, grid_y]
        indices, anch = self.find_3_positive(predictions, targets)

        matching_bs = [[] for _ in predictions]
        matching_as = [[] for _ in predictions]
        matching_gjs = [[] for _ in predictions]
        matching_gis = [[] for _ in predictions]
        matching_targets = [[] for _ in predictions]
        matching_anchs = [[] for _ in predictions]

        # label assignment for each image
        for batch_idx in range(predictions[0].shape[0]):

            # targets of this image
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 1:5]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_age = []
            p_race = []
            p_masked = []
            p_skintone = []
            p_emotion = []
            p_gender = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, map in enumerate(predictions):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = map[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_age.append(fg_pred[:, 5:11])
                p_race.append(fg_pred[:, 11:14])
                p_masked.append(fg_pred[:, 14:16])
                p_skintone.append(fg_pred[:, 16:20])
                p_emotion.append(fg_pred[:, 20:27])
                p_gender.append(fg_pred[:, 27:29])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.strides[i]  # / 8.
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.strides[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue

            p_obj = torch.cat(p_obj, dim=0)
            p_age = torch.cat(p_age, dim=0)
            p_race = torch.cat(p_race, dim=0)
            p_masked = torch.cat(p_masked, dim=0)
            p_skintone = torch.cat(p_skintone, dim=0)
            p_emotion = torch.cat(p_emotion, dim=0)
            p_gender = torch.cat(p_gender, dim=0)
            
            from_which_layer = torch.cat(from_which_layer, dim=0).to(this_target.device)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            # Cost matrix
            pair_wise_iou = bboxes_iou(txyxy, pxyxys)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_age_per_image      = F.one_hot(this_target[:, 5].to(torch.int64), 6).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            gt_race_per_image     = F.one_hot(this_target[:, 6].to(torch.int64), 3).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            gt_masked_per_image   = F.one_hot(this_target[:, 7].to(torch.int64), 2).float().float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            gt_skintone_per_image = F.one_hot(this_target[:, 8].to(torch.int64), 4).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            gt_emotion_per_image  = F.one_hot(this_target[:, 9].to(torch.int64), 7).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            gt_gender_per_image   = F.one_hot(this_target[:, 10].to(torch.int64), 2).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            num_gt = this_target.shape[0]

            age_preds_      = p_age.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            race_preds_     = p_race.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            masked_preds_   = p_masked.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            skintone_preds_ = p_skintone.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            emotion_preds_  = p_emotion.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            gender_preds_   = p_gender.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()

            age_y      = age_preds_.sqrt_()
            race_y     = race_preds_.sqrt_()
            masked_y   = masked_preds_.sqrt_()
            skintone_y = skintone_preds_.sqrt_()
            emotion_y  = emotion_preds_.sqrt_()
            gender_y   = gender_preds_.sqrt_()

            pair_wise_age_loss      = F.binary_cross_entropy_with_logits(torch.log(age_y / (1 - age_y)), gt_age_per_image, reduction="none").sum(-1)
            pair_wise_race_loss     = F.binary_cross_entropy_with_logits(torch.log(race_y / (1 - race_y)), gt_race_per_image, reduction="none").sum(-1)
            pair_wise_masked_loss   = F.binary_cross_entropy_with_logits(torch.log(masked_y / (1 - masked_y)), gt_masked_per_image, reduction="none").sum(-1)
            pair_wise_skintone_loss = F.binary_cross_entropy_with_logits(torch.log(skintone_y / (1 - skintone_y)), gt_skintone_per_image, reduction="none").sum(-1)
            pair_wise_emotion_loss  = F.binary_cross_entropy_with_logits(torch.log(emotion_y / (1 - emotion_y)), gt_emotion_per_image, reduction="none").sum(-1)
            pair_wise_gender_loss   = F.binary_cross_entropy_with_logits(torch.log(gender_y / (1 - gender_y)), gt_gender_per_image, reduction="none").sum(-1)

            del age_preds_
            del race_preds_
            del masked_preds_
            del skintone_preds_
            del emotion_preds_
            del gender_preds_

            cost = (
                (pair_wise_age_loss
                + pair_wise_race_loss
                + pair_wise_masked_loss
                + pair_wise_skintone_loss
                + pair_wise_emotion_loss
                + pair_wise_gender_loss) / 6.0
                + 5.0 * pair_wise_iou_loss
            )

            # Dynamic k
            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(self.nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(self.nl):
            if matching_targets[i]:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([]).type_as(targets)
                matching_as[i] = torch.tensor([]).type_as(targets)
                matching_gjs[i] = torch.tensor([]).type_as(targets)
                matching_gis[i] = torch.tensor([]).type_as(targets)
                matching_targets[i] = torch.tensor([]).type_as(targets)
                matching_anchs[i] = torch.tensor([]).type_as(targets)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, predictions, targets):
        """
        Args:
            predictions(tensor): [nb, na, w, h, ch]
            targets(tensor): [image_idx, class, x, y, w, h]
        Return:
            indice: [img_idx, anchor_idx, grid_x, grid_y]
            anchor: [anchor_w, anchor_h]
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(12).type_as(targets).long()  # normalized to gridspace gain
        ai = torch.arange(na).type_as(targets).view(na, 1).repeat(1, nt)  # [na, nt, 1]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # [na, nt, 7]

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):

            # put anchor and target to feature map
            anchors = (self.anchors[i] / self.strides[i]).type_as(predictions[i])
            gain[1:5] = self.strides[i]
            target = targets / gain

            # TODO: check this ([3, 2, 3, 2] or [2, 3, 2, 3] ?)
            gain[1:5] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]  # w and h

            # Match targets to anchors
            if nt:
                # target and anchor wh ratio in threshold
                r = target[:, :, 3:5] / anchors[:, None]  # wh ratio
                wh_mask = torch.max(r, 1. / r).max(2)[0] < self.threshold  # compare
                t = target[wh_mask]

                # Positive adjacent grid
                gxy = t[:, 1:3]  # grid xy
                gxi = gain[[1, 2]] - gxy  # inverse grid xy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, 0].long(), t[:, 5:].long()  # image_idx, class
            gxy = t[:, 1:3]  # grid xy
            gwh = t[:, 3:5]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 11].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[2] - 1), gi.clamp_(0, gain[1] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    box2 = box2.T

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
