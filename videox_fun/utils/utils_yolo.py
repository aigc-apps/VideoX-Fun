import os
import cv2
import numpy as np
from pathlib import Path

import cv2
import numpy as np
import requests

# ultralytics builds a module-level `SettingsManager()` at import time that calls
# `dist.barrier(device_ids=[RANK])`, passing the GLOBAL rank as a CUDA device ordinal
# (ultralytics/utils/__init__.py -> `torch_distributed_zero_first(RANK)`; note it should have
# used LOCAL_RANK). On multi-node runs the global rank can exceed the per-node GPU count
# (e.g. rank 15 on an 8-GPU node), which crashes the import with "CUDA error: invalid device
# ordinal". Present the node-local rank to ultralytics just for this import so its barrier
# targets a valid device, then restore the true RANK so distributed training is unaffected.
_REAL_RANK = os.environ.get("RANK")
if _REAL_RANK is not None:
    os.environ["RANK"] = os.environ.get("LOCAL_RANK", "0")
try:
    from ultralytics import YOLO
finally:
    if _REAL_RANK is not None:
        os.environ["RANK"] = _REAL_RANK


MODEL_DIR = "./models"
instance_url = "https://zhoumo-bj.oss-cn-beijing.aliyuncs.com/Code/model_data/Stable_Diffusion/annotator/yolov8x-seg.pt"
yolo26_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt"

def urldownload(url, filename):
    """
    Download a file to the specified directory.
    :param url: URL of the file to download.
    :param filename: Destination path, e.g. ./test.xls.
    """
    down_res = requests.get(url)
    with open(filename,'wb') as file:
        file.write(down_res.content)

class ObjectInstanceDetector:
    def __init__(self, model_dir=MODEL_DIR, combine_overlap_boxes=False, device=None):
        model_path = os.path.join(model_dir, "yolov8x-seg.pt")
        if not os.path.exists(model_path):
            urldownload(instance_url, model_path)

        self.model  = YOLO(model_path).to(device)
        self.cob    = combine_overlap_boxes
        self.overlap_threhold = 0.05
    
    def cal_miou(self, mask1, mask2):
        tp = mask1 * mask2
        if np.shape(mask2)[0] != np.shape(mask2)[1]:
            tile_mask1 = np.tile(mask1, [np.shape(mask2)[0], 1, 1])
        tp_fp_fn = np.clip(tile_mask1 + mask2, 0, 1)
        miou = np.sum(np.sum(tp, 1), 1) / np.sum(np.sum(tp_fp_fn, 1), 1) 
        return miou

    def combine_overlap_boxes(self, boxes, masks):       
        outputs         = []
        outputs_masks   = []
        while np.shape(boxes)[0]:
            # Get the mask dimensions of the smallest box
            h, w = np.shape(masks[0])

            # Store the smallest box coordinates and mask
            outputs.append(boxes[0:1, :])
            outputs_masks.append(masks[0:1])
            if len(boxes) == 1:
                break
            
            # Compute mIoU between the smallest box and remaining boxes
            outputs_mask    = np.zeros([1, h, w])
            outputs_mask[0, int(outputs[-1][0][1]):int(outputs[-1][0][3]), int(outputs[-1][0][0]):int(outputs[-1][0][2])] = 1
            miou            = self.cal_miou(outputs_mask, masks[1:])

            # Boxes with mIoU above threshold are considered significantly overlapping
            overlap_boxes   = boxes[1:][miou > self.overlap_threhold]
            overlap_masks   = masks[1:][miou > self.overlap_threhold]
            if len(overlap_boxes) > 0:
                overlap_boxes   = np.concatenate([outputs[-1], overlap_boxes], 0)
                overlap_masks   = np.concatenate([outputs_masks[-1], overlap_masks], 0)

                outputs[-1]         = np.array([[np.min(overlap_boxes[:, 0]), np.min(overlap_boxes[:, 1]), np.max(overlap_boxes[:, 2]), np.max(overlap_boxes[:, 3])]])
                outputs_masks[-1]   = np.max(overlap_masks, 0, keepdims=True)
                
                cv2.imwrite("test.jpg", overlap_masks[0] * 255)

            # Keep non-overlapping masks and boxes
            masks           = masks[1:][miou <= self.overlap_threhold]
            boxes           = boxes[1:][miou <= self.overlap_threhold]

            if len(boxes) == 0:
                break

            # Re-compute mIoU against remaining boxes
            outputs_mask    = np.zeros([1, h, w])
            outputs_mask[0, int(outputs[-1][0][1]):int(outputs[-1][0][3]), int(outputs[-1][0][0]):int(outputs[-1][0][2])] = 1
            miou            = self.cal_miou(outputs_mask, masks)

            # If any remaining boxes overlap significantly, merge them
            overlap_boxes   = boxes[miou > self.overlap_threhold]
            if len(overlap_boxes) > 0:
                boxes       = np.concatenate([outputs[-1], boxes], 0)
                masks       = np.concatenate([outputs_masks[-1], masks], 0)
                del outputs[-1]
                del outputs_masks[-1]
        # Stack results
        if len(outputs) != 0:
            outputs = np.concatenate(outputs, 0)
            outputs_masks = np.concatenate(outputs_masks, 0)
        return outputs, outputs_masks

    def __call__(self, img):
        h, w, c     = np.shape(img)
        results     = self.model.predict(source=img, imgsz=1024, conf=0.40)

        try:
            categories  = np.array([result.boxes.cls.cpu().numpy() for result in results])[0]
            boxes       = np.array([result.boxes.xyxy.cpu().numpy() for result in results])[0]
            masks       = np.array([result.masks.data.cpu().numpy() for result in results])[0]
        except Exception:
            return [], [], []
        
        if len(boxes) != 0:
            if self.cob:
                masks       = np.array([cv2.resize(mask, (w, h), cv2.INTER_NEAREST) for mask in masks])

                areas       = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                args        = np.argsort(areas)
                boxes       = boxes[args]
                masks       = masks[args]
                
                outputs, outputs_masks = self.combine_overlap_boxes(boxes, masks)
                outputs, outputs_masks = self.combine_overlap_boxes(outputs, outputs_masks)

                areas       = (outputs[:, 3] - outputs[:, 1]) * (outputs[:, 2] - outputs[:, 0])
                args        = np.argsort(areas)[::-1]
                outputs       = outputs[args]
                outputs_masks = outputs_masks[args]

                outputs_masks = np.uint8(outputs_masks) * 255 
                outputs_masks = [cv2.resize(mask, (w, h), cv2.INTER_NEAREST) for box, mask in zip(outputs, outputs_masks)]
                return outputs, boxes, outputs_masks
            else:
                areas       = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                args        = np.argsort(areas)[::-1]

                boxes       = boxes[args]
                masks       = masks[args]

                masks       = np.uint8(masks) * 255 
                masks       = [cv2.resize(mask, (w, h), cv2.INTER_NEAREST) for box, mask in zip(boxes, masks)]
                return boxes, boxes, masks
        else:
            return [], boxes, []


class ObjectDetector:
    """
    COCO-pretrained object detection and segmentation based on YOLO26-seg.
    Supports 80 COCO categories. Prediction results include class names.
    """

    def __init__(self, model_dir=MODEL_DIR, model_size="x", device=None):
        """
        Initialize YOLO26 COCO detector.

        Args:
            model_dir: Directory to store model weights.
            model_size: Model scale, one of "n", "s", "m", "l", "x".
            device: Inference device, e.g. "cuda:0" or "cpu". Auto-selects if None.
        """
        model_name = f"yolo26{model_size}-seg.pt"
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_dir, exist_ok=True)
            urldownload(yolo26_url.replace("yolo26x-seg.pt", model_name), model_path)

        self.model = YOLO(model_path)
        if device is not None:
            self.model.to(device)

    def __call__(self, img, conf=0.25, imgsz=1024):
        """
        Run COCO 80-class object detection on the input image.

        Args:
            img: Input image (numpy array, BGR format) or image file path.
            conf: Confidence threshold, default 0.25.
            imgsz: Inference image size, default 1024.

        Returns:
            list[dict]: Each detection result contains:
                - "class_name" (str): COCO category name
                - "class_id" (int): Category ID
                - "confidence" (float): Confidence score
                - "bbox" (np.ndarray): Bounding box [x1, y1, x2, y2]
                - "mask" (np.ndarray | None): Segmentation mask (H, W), uint8 format
        """
        # Get original image dimensions
        if isinstance(img, str):
            img_bgr = cv2.imread(img)
        else:
            img_bgr = img
        h, w = img_bgr.shape[:2]

        results = self.model.predict(source=img_bgr, imgsz=imgsz, conf=conf)

        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Get class name mapping (COCO 80 classes)
            id_to_name = {int(k): v for k, v in result.names.items()} if result.names else {}

            # Extract segmentation masks if available
            has_masks = result.masks is not None and result.masks.data is not None
            if has_masks:
                mask_data = result.masks.data.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                class_id = int(cls_ids[i])
                class_name = id_to_name.get(class_id, str(class_id))
                confidence = float(confs[i])
                bbox = boxes_xyxy[i]

                # Resize mask to original image dimensions
                if has_masks and i < len(mask_data):
                    mask = mask_data[i]
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = None

                detections.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": bbox,
                    "mask": mask,
                })

        # Sort by confidence in descending order
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections

    def detect_and_draw(self, img, conf=0.25, imgsz=1024):
        """
        Detect objects and draw results on the image (for debugging/visualization).

        Args:
            img: Input image (numpy array, BGR format) or image file path.
            conf: Confidence threshold.
            imgsz: Inference image size.

        Returns:
            tuple: (annotated_image, detections)
        """
        if isinstance(img, str):
            img_bgr = cv2.imread(img)
        else:
            img_bgr = img.copy()

        detections = self(img_bgr, conf=conf, imgsz=imgsz)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"].astype(int)
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if det["mask"] is not None:
                colored_mask = np.zeros_like(img_bgr)
                colored_mask[:, :, 1] = det["mask"]
                img_bgr = cv2.addWeighted(img_bgr, 1.0, colored_mask, 0.4, 0)

        return img_bgr, detections
