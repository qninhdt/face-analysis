from typing import List

import time
import numpy as np
import cv2
import onnxruntime as rt

from utils.bbox import cxcywh2xyxy


class FaceDetector:
    def __init__(
        self, model_path: str, conf_thre=0.6, nms_thre=0.35, device: str = "cpu"
    ):
        self.session = rt.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"] if device == "cuda" else None,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.device = device
        self.image_size = 640

    def preprocess(self, img: np.ndarray):
        h, w, _ = img.shape

        img = img.copy()

        # pad to square
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            img = cv2.copyMakeBorder(
                img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            img = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # resize to 640x640
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

        # normalize
        mean = np.array([0.6596, 0.6235, 0.5875])
        std = np.array([0.2272, 0.2248, 0.2345])

        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))

        return img.astype(np.float32), (w, h)

    def detect(self, input: List[np.ndarray] | np.ndarray):
        last = time.time()

        if isinstance(input, np.ndarray):
            input = [input]

        origin_sizes = []
        preprocessed = []
        for i in range(len(input)):
            img, size = self.preprocess(input[i])
            preprocessed.append(img)
            origin_sizes.append(size)
        preprocessed = np.array(preprocessed)

        # inference
        output = self.session.run([self.output_name], {self.input_name: preprocessed})

        # postprocess
        output = self.postprocess(output[0], origin_sizes)

        total = time.time() - last
        print(f"Total inference time: {total:.4f}s")

        return output

    def detect_with_result_images(self, input: List[np.ndarray] | np.ndarray):
        if isinstance(input, np.ndarray):
            input = [input]

        outputs = self.detect(input)
        # draw bounding boxes
        results = []
        for i in range(len(input)):
            # copy to draw on
            image = input[i].copy()

            for box in outputs[i]:
                box = box["box"]
                box[0] -= box[2] / 2
                box[1] -= box[3] / 2
                box[2] += box[0]
                box[3] += box[1]
                cv2.rectangle(
                    image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    2,
                )

            results.append(image)

        return outputs, results

    def postprocess(self, predictions, origin_sizes):
        output = [[] for _ in range(predictions.shape[0])]

        for i in range(predictions.shape[0]):
            image_pred = predictions[i]

            # Get class and correspond score
            age_pred = np.argmax(image_pred[:, 5:11], axis=1, keepdims=True)
            race_pred = np.argmax(image_pred[:, 11:14], axis=1, keepdims=True)
            masked_pred = np.argmax(image_pred[:, 14:16], axis=1, keepdims=True)
            skintone_pred = np.argmax(image_pred[:, 16:20], axis=1, keepdims=True)
            emotion_pred = np.argmax(image_pred[:, 20:27], axis=1, keepdims=True)
            gender_pred = np.argmax(image_pred[:, 27:29], axis=1, keepdims=True)
            box_pred = image_pred[:, :4]

            confidence = image_pred[:, 4]
            conf_mask = confidence >= self.conf_thre

            detections = np.concatenate(
                (
                    box_pred,
                    confidence[:, np.newaxis],
                    age_pred.astype(float),
                    race_pred.astype(float),
                    masked_pred.astype(float),
                    skintone_pred.astype(float),
                    emotion_pred.astype(float),
                    gender_pred.astype(float),
                ),
                axis=1,
            )

            detections = detections[conf_mask]

            if detections.shape[0] > 100:
                detections = detections[:100]

            nms_out_index = self.non_maximum_suppression(
                cxcywh2xyxy(detections[:, :4]), detections[:, 4], self.nms_thre
            )
            detections = detections[nms_out_index]

            # convert to original size
            if origin_sizes is not None:
                size = origin_sizes[i]
                scale = np.array(
                    [max(size[0], size[1])] * 4,
                    dtype=np.float32,
                )
                detections[:, :4] = detections[:, :4] * (scale / self.image_size)

                for box in detections[:, :4]:
                    if size[0] < size[1]:
                        box[0] -= (size[1] - size[0]) / 2
                    else:
                        box[1] -= (size[0] - size[1]) / 2

            output[i] = [
                {
                    "box": detections[i, :4],
                    "score": detections[i, 4],
                    "age": detections[i, 5],
                    "race": detections[i, 6],
                    "masked": detections[i, 7],
                    "skintone": detections[i, 8],
                    "emotion": detections[i, 9],
                    "gender": detections[i, 10],
                }
                for i in range(detections.shape[0])
            ]

        return output

    def non_maximum_suppression(self, boxes, scores, threshold):
        # return mask of indices of kept boxes

        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes are integers, convert them to floats
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the score/probability of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)

        # keep looping while some indexes still remain in the indexes list

        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the intersection

            xx1_int = np.maximum(x1[i], x1[idxs[:last]])
            yy1_int = np.maximum(y1[i], y1[idxs[:last]])
            xx2_int = np.minimum(x2[i], x2[idxs[:last]])
            yy2_int = np.minimum(y2[i], y2[idxs[:last]])

            ww_int = np.maximum(0, xx2_int - xx1_int + 1)
            hh_int = np.maximum(0, yy2_int - yy1_int + 1)

            area_int = ww_int * hh_int

            # find the union
            area_union = area[i] + area[idxs[:last]] - area_int

            # compute the ratio of overlap
            overlap = area_int / area_union

            # delete all indexes from the index list that have
            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > threshold)[0]))
            )

        # return only the bounding boxes that were picked
        return pick

    def calculate_iou(self, box, boxes):
        # Calculate intersection area
        intersection_x1 = np.maximum(box[0], boxes[:, 0])
        intersection_y1 = np.maximum(box[1], boxes[:, 1])
        intersection_x2 = np.minimum(box[2], boxes[:, 2])
        intersection_y2 = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(
            0, intersection_x2 - intersection_x1 + 1
        ) * np.maximum(0, intersection_y2 - intersection_y1 + 1)

        # Calculate area of the boxes
        area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

        # Calculate union area
        union_area = area_box + area_boxes - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou
