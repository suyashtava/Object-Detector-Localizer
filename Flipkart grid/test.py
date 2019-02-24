from train import *
import cv2
import glob
import numpy as np
from PIL import Image

WEIGHTS_FILE = "model-0.90.h5"

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.2
MAX_OUTPUT_SIZE = 1

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    with open('test.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)

    for i in range(1, len(lines)):
        filename = lines[i][0]
        img = Image.open(filename)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        feat_scaled = preprocess_input(np.array(img, dtype=np.float32))

        pred = np.squeeze(model.predict(feat_scaled[np.newaxis,:]))
        height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]

        coords = np.arange(pred.shape[0] * pred.shape[1])
        y = (y_f + coords // pred.shape[0]) / (pred.shape[0] - 1)
        x = (x_f + coords % pred.shape[1]) / (pred.shape[1] - 1)

        boxes = np.stack([y, x, height, width, score], axis=-1)
        boxes = boxes[np.where(boxes[...,-1] >= SCORE_THRESHOLD)]

        selected_indices = tf.image.non_max_suppression(boxes[...,:-1], boxes[...,-1], MAX_OUTPUT_SIZE, IOU_THRESHOLD)
        selected_indices = tf.Session().run(selected_indices)

        for y_c, x_c, h, w, _ in boxes[selected_indices]:
            x0 = 640 * (x_c - w / 2)
            y0 = 480 * (y_c - h / 2)
            x1 = x0 + 640 * w
            y1 = y0 + 480 * h

            lines[i][1] = int(x0)
            lines[i][2] = int(x1)
            lines[i][3] = int(y0)
            lines[i][4] = int(y1)

    with open('test.csv', 'w', newline = '') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)

    readFile.close()
    writeFile.close()


if __name__ == "__main__":
    main()