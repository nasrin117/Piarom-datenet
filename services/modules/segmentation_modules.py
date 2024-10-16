import cv2
import math
import numpy as np

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def extract_boxes(box_predictions, image_height, image_width, input_height, input_width):
    # Extract boxes from predictions
    boxes = box_predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes,
                                (input_height, input_width),
                                (image_height, image_width))

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    # Check the boxes are within the image
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)

    return boxes

def rescale_boxes(boxes, input_shape, image_shape):
    # Rescale boxes to original image dimensions
    input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

    return boxes

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def masks2segments(masks, strategy='largest'):
    segments = []
    for x in masks.astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., 0] -= pad[0]  # x padding
    coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords

def clip_coords(coords, shape):
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y

def get_points(outputs, image_height, image_width, input_height, input_width, conf_threshold, iou_threshold):
    box_output  = outputs[0]
    predictions = np.squeeze(box_output).T
    num_classes = box_output.shape[1] - 32 - 4

    scores = np.max(predictions[:, 4:4+num_classes], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    box_predictions = predictions[..., :num_classes+4]
    mask_predictions = predictions[..., num_classes+4:]

    class_ids = np.argmax(box_predictions[:, 4:], axis=1)

    boxes = extract_boxes(box_predictions, image_height, image_width, input_height, input_width)

    indices = nms(boxes, scores, iou_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    mask_predictions = mask_predictions[indices]

    mask_output = np.squeeze(outputs[1])

    num_mask, mask_height, mask_width = mask_output.shape
    masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
    masks = masks.reshape((-1, mask_height, mask_width))
    scale_boxes = rescale_boxes(boxes, (image_height, image_width), (mask_height, mask_width))

    mask_maps = np.zeros((len(scale_boxes), image_height, image_width))

    for i in range(len(scale_boxes)):
        scale_x1 = int(math.floor(scale_boxes[i][0]))
        scale_y1 = int(math.floor(scale_boxes[i][1]))
        scale_x2 = int(math.ceil(scale_boxes[i][2]))
        scale_y2 = int(math.ceil(scale_boxes[i][3]))

        x1 = int(math.floor(boxes[i][0]))
        y1 = int(math.floor(boxes[i][1]))
        x2 = int(math.ceil(boxes[i][2]))
        y2 = int(math.ceil(boxes[i][3]))

        scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
        crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
        
        # crop_mask = cv2.blur(crop_mask, blur_size)
        crop_mask = (crop_mask > 0.5).astype(np.uint8)
        mask_maps[i, y1:y2, x1:x2] = crop_mask

    points = [scale_coords(mask_maps.shape[1:], x, (image_height, image_width), normalize=False) for x in masks2segments(mask_maps)]

    return points
