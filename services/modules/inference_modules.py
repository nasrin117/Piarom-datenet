import os
import cv2
import numpy as np

def resize_with_padding(img, expected_size):
    height, width = img.shape[:2]
    scale = expected_size[0] / max(width, height)
    new_width, new_height = int(width * scale), int(height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))
    delta_width = expected_size[0] - new_width
    delta_height = expected_size[1] - new_height
    padding = [(delta_width // 2, delta_height // 2), (delta_width - delta_width // 2, delta_height - delta_height // 2)]
    padded_img = cv2.copyMakeBorder(resized_img, padding[0][1],padding[1][1], padding[0][0], padding[1][0], cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    return padded_img

def put_label(image, text, color, pt):
    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)[0]
    image = cv2.rectangle(
        img=image, 
        rec=(pt[0],pt[1]+0, text_width, text_height+10), 
        color=color, 
        thickness=-1
        )
    image = cv2.putText(
        img=image,
        text=text,
        org=(pt[0], pt[1]+15),
        fontScale=0.6,
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        color=(0,0,0),
        thickness=1
        )
    
    return image

def find_extreme_points(contour):
    min_x, max_x = np.min(contour[:, 0]), np.max(contour[:, 0])
    min_y, max_y = np.min(contour[:, 1]), np.max(contour[:, 1])

    top_left = contour[np.argmin(np.linalg.norm(contour - [min_x, min_y], axis=1))]
    top_right = contour[np.argmin(np.linalg.norm(contour - [max_x, min_y], axis=1))]
    bot_left = contour[np.argmin(np.linalg.norm(contour - [min_x, max_y], axis=1))]
    bot_right = contour[np.argmin(np.linalg.norm(contour - [max_x, max_y], axis=1))]

    return {
        "top_left": tuple(top_left),
        "top_right": tuple(top_right),
        "bot_left": tuple(bot_left),
        "bot_right": tuple(bot_right),
    }

def crop_image(image, imgw, paddig=True):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint64))
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    points = find_extreme_points(cnts_sorted[0].squeeze(axis=1))
    points = np.float32([points["top_left"], points["top_right"], points["bot_left"], points["bot_right"]])

    width = imgw
    height = int(width * 1.40625)

    dst_points = np.float32([[0, 0], [width, 0],[0, height], [width, height]]) 
    matrix = cv2.getPerspectiveTransform(points, dst_points)
    img = cv2.warpPerspective(img, matrix, (width, height))

    if paddig:
        img = resize_with_padding(img, (imgw, imgw))

    return img

def classify_onnx_predict(image, crop_image, imgsz, ort_session, class_names, pt):
    max_size = max(crop_image.shape[0], crop_image.shape[1])
    onnx_img = resize_with_padding(crop_image, (max_size, max_size))
    onnx_img = cv2.cvtColor(onnx_img, cv2.COLOR_BGR2RGB)
    onnx_img = cv2.resize(onnx_img, (imgsz, imgsz)) / 255.0
    onnx_img = np.transpose(onnx_img, (2, 0, 1))
    onnx_img = np.expand_dims(onnx_img, axis=0).astype(np.float32)

    result = ort_session.run(None, {ort_session.get_inputs()[0].name: onnx_img})

    for i in range(result[0].shape[0]):
        pred = result[0][i]
        class_num = np.argmax(pred)
        class_name = class_names[class_num]

        if "_gold" in class_name:
            color = "gold"
        elif "_black" in class_name:
            color = "black"
        else:
            color = "None"

        size = (int(crop_image.shape[1] * 320 / 1280), int(crop_image.shape[0] * 450 / (1280*1.40625)))

        
        image = put_label(image, class_name, (0, 0, 255), pt)


        return class_name, color, size

def detection_onnx_predict(image, imgw, ort_session):
    onnx_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    onnx_img = cv2.resize(onnx_img, (imgw, imgw)) / 255.0
    onnx_img = np.transpose(onnx_img, (2, 0, 1))
    onnx_img = np.expand_dims(onnx_img, axis=0).astype(np.float32)

    result = ort_session.run(None, {ort_session.get_inputs()[0].name: onnx_img})

    return result

def sort_boxes(boxes):
    boxes = np.transpose(boxes)
    boxes = sorted(boxes, key=lambda box: box[1], reverse=True)
    boxes = np.array_split(boxes, 10)
    boxes = [sorted(inner_list, key=lambda x: float(x[0]), reverse=True) for inner_list in boxes]
    boxes = np.concatenate(boxes)
    boxes = np.transpose(boxes)

    return boxes

def classify_bboxes(model_output, classify_model, image, image_path, imgw, imgsz, class_names, show=False, save=False):
    bboxes = []
    croped_images = []
    pr_class_names = []
    colors = []
    sizes = []

    for i in range(model_output[0].shape[0]):
        image_copy = image.copy()
        height_coefficient, width_coefficient = image.shape[0]/imgw, image.shape[1]/imgw

        pred = sort_boxes(model_output[0][i])
        anchor_scores = np.max(pred[4:, :], axis=0)
        bbox_coords = np.transpose(pred[:4])
        keep_indices = cv2.dnn.NMSBoxes(bbox_coords, anchor_scores, 0.25, 0.4)

        for idx in sorted(keep_indices, reverse=True):
            x, y, w, h = bbox_coords[idx].astype(np.uint64)
            x1, y1 = int((x - w/2)*width_coefficient), int((y - h/2)*height_coefficient)
            x2, y2 = int((x + w/2)*width_coefficient), int((y + h/2)*height_coefficient)
            crop_image = image_copy[y1:y2, x1:x2]
            class_name, color, size = classify_onnx_predict(
                image=image,
                crop_image=crop_image,
                imgsz=imgsz, 
                ort_session=classify_model, 
                class_names=class_names, 
                pt=(x1, y2)
                )
            bboxes.append([x1, y1, x2, y2])
            croped_images.append(crop_image)
            pr_class_names.append(class_name)
            colors.append(color)
            sizes.append(size)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if save:
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(f"output/{image_path.split('/')[-1]}", image)

        if show:
            cv2.imshow("_".join(image_path.split("/")[-1].split("_")[:-1]), image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return bboxes, croped_images, pr_class_names, colors, sizes, image
