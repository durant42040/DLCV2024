from PIL import Image
import numpy as np
import argparse
import os


def mse(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    img1_array = np.array(image1).astype(np.float32)
    img2_array = np.array(image2).astype(np.float32)

    mse_value = np.mean((img1_array - img2_array) ** 2)
    return mse_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, help="Path to input noise tensors.")
    parser.add_argument("--gt", type=str, help="Path to input noise tensors.")
    args = parser.parse_args()

    pred = os.listdir(args.pred)
    gt = os.listdir(args.gt)

    pred.sort()
    gt.sort()

    assert len(pred) == len(gt)
    pred_images = [os.path.join(args.pred, f) for f in pred]
    gt_images = [os.path.join(args.gt, f) for f in gt]

    mse_values = [mse(pred_images[i], gt_images[i]) for i in range(len(pred_images))]

    print(np.mean(mse_values))
