import torch
import torchvision
import numpy as np
from utils.bbox import cxcywh2xyxy, box_iou

IMAGE_SIZE = 640

def postprocess(predictions, conf_thre=0.6, nms_thre=0.35, origin_sizes=None):
    max_det = 300  # maximum number of detections per image
    max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()

    output = [None for _ in range(predictions.shape[0])]
    for i in range(predictions.shape[0]):
        image_pred = predictions[i]     

        # Get class and correspond score
        age_conf, age_pred = torch.max(image_pred[:, 5:11], 1, keepdim=True)
        race_conf, race_pred = torch.max(image_pred[:, 11:14], 1, keepdim=True)
        masked_conf, masked_pred = torch.max(image_pred[:, 14:16], 1, keepdim=True)
        skintone_conf, skintone_pred = torch.max(image_pred[:, 16:20], 1, keepdim=True)
        emotion_conf, emotion_pred = torch.max(image_pred[:, 20:27], 1, keepdim=True)
        gender_conf, gender_pred = torch.max(image_pred[:, 27:29], 1, keepdim=True)
        box_pred = image_pred[:, :4]
        
        confidence = image_pred[:, 4]
        conf_mask = (confidence >= conf_thre).squeeze()
        
        # Detections ordered as (x1, y1, x2, y2, confidence, class_pred)
        detections = torch.cat((image_pred[:, :4], confidence.unsqueeze(-1), 
                                age_pred.float(),
                                race_pred.float(),
                                masked_pred.float(),
                                skintone_pred.float(),
                                emotion_pred.float(),
                                gender_pred.float()), 1)
        
        detections = detections[conf_mask]

        if detections.shape[0] > max_nms:
            detections = detections[:max_nms]
        # if not detections.size(0):
        #     continue

        nms_out_index = torchvision.ops.batched_nms(
            cxcywh2xyxy(detections[:, :4]),
            detections[:, 4],
            torch.zeros_like(detections[:, 5]),
            nms_thre,
        )

        detections = detections[nms_out_index]
        if detections.shape[0] > max_det:  # limit detections
            detections = detections[:max_det]

        # convert to original size
        if origin_sizes is not None:
            size = origin_sizes[i]
            scale = torch.tensor([max(size[0], size[1])] * 4, device=detections.device, dtype=torch.float32)
            detections[:, :4] = detections[:, :4] * (scale / IMAGE_SIZE)

            for box in detections[:, :4]:

                if size[0] < size[1]:
                    box[0] -= (size[1] - size[0]) / 2
                else:
                    box[1] -= (size[0] - size[1]) / 2

        output[i] = {
            "boxes": detections[:, :4],
            "scores": detections[:, 4],
            "age": detections[:, 5],
            "race": detections[:, 6],
            "masked": detections[:, 7],
            "skintone": detections[:, 8],
            "emotion": detections[:, 9],
            "gender": detections[:, 10],
        }

    return output

def match_pred_boxes(preds, targets):
    # return [nb, n_pred, (pred_idx, target_idx)]

    results = []

    nb = len(preds)

    # match all preds with ioa > 0.5
    for i in range(nb):
        pred_boxes = preds[i]['boxes']
        target_boxes = targets[i]['boxes']

        # convert to cxcywh to xyxy
        pred_boxes = cxcywh2xyxy(pred_boxes)
        target_boxes = cxcywh2xyxy(target_boxes)

        # calculate iou
        iou = box_iou(pred_boxes, target_boxes)

        # get max iou
        max_iou, max_idx = iou.max(1)

        # get pred_idx
        pred_idx = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)

        # remove pred_idx with max_idx = -1
        mask = max_iou > 0.5
        pred_idx = pred_idx[mask]
        max_idx = max_idx[mask]
        
        # append to results
        results.append(torch.stack([pred_idx, max_idx], 1))

    return results

# def demo_postprocess(predictions, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
#     max_det = 300  # maximum number of detections per image
#     max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()

#     output = [None for _ in range(predictions.shape[0])]
#     for i in range(predictions.shape[0]):
#         image_pred = predictions[i]
#         # If none are remaining => process next image
#         if not image_pred.shape[0]:
#             continue
#         # Get class and correspond score
#         class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
#         confidence = image_pred[:, 4] * class_conf.squeeze()
#         conf_mask = (confidence >= conf_thre).squeeze()
#         # Detections ordered as (x1, y1, x2, y2, confidence, class_pred)
#         detections = torch.cat((image_pred[:, :4], confidence.unsqueeze(-1), class_pred.float()), 1)
#         detections = detections[conf_mask]
#         if detections.shape[0] > max_nms:
#             detections = detections[:max_nms]
#         if not detections.size(0):
#             continue

#         if class_agnostic:
#             nms_out_index = torchvision.ops.nms(
#                 detections[:, :4],
#                 detections[:, 4],
#                 nms_thre,
#             )
#         else:
#             nms_out_index = torchvision.ops.batched_nms(
#                 detections[:, :4],
#                 detections[:, 4],
#                 torch.zeros_like(detections[:, 5]),
#                 nms_thre,
#             )

#         detections = detections[nms_out_index]
#         if detections.shape[0] > max_det:  # limit detections
#             detections = detections[:max_det]
#         output[i] = detections

#     return output


# def format_outputs(outputs, ids, hws, val_size, class_ids, labels):
#     """
#     outputs: [batch, [x1, y1, x2, y2, confidence, class_pred]]
#     """

#     json_list = []
#     # det_list (list[list]): shape(num_images, num_classes)
#     det_list = [[np.empty(shape=[0, 5]) for _ in range(len(class_ids))] for _ in range(len(outputs))]

#     # for each image
#     for i, (output, img_h, img_w, img_id) in enumerate(zip(outputs, hws[0], hws[1], ids)):
#         if output is None:
#             continue

#         bboxes = output[:, 0:4]
#         scale = min(val_size[0] / float(img_w), val_size[1] / float(img_h))
#         bboxes /= scale
#         coco_bboxes = xyxy2xywh(bboxes)

#         scores = output[:, 4]
#         clses = output[:, 5]

#         # COCO format follows the prediction
#         for bbox, cocobox, score, cls in zip(bboxes, coco_bboxes, scores, clses):
#             # COCO format
#             cls = int(cls)
#             class_id = class_ids[cls]
#             pred_data = {
#                 "image_id": int(img_id),
#                 "category_id": class_id,
#                 "bbox": cocobox.cpu().numpy().tolist(),
#                 "score": score.cpu().numpy().item(),
#                 "segmentation": [],
#             }
#             json_list.append(pred_data)

#         # VOC format follows the class
#         for c in range(len(class_ids)):
#             # detection np.array(x1, y1, x2, y2, score)
#             det_ind = clses == c
#             detections = output[det_ind, 0:5]
#             det_list[i][c] = detections.cpu().numpy()

#     return json_list, det_list
