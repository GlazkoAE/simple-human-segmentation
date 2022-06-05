import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from dataset import colour_code_segmentation, reverse_one_hot, get_preprocessing


def blur(original_image, kernel=3, factor=10):
    (h, w) = original_image.shape[:2]
    k_w = int(w / kernel)
    k_h = int(h / kernel)

    # ensure the width of the kernel is odd
    if k_w % 2 == 0:
        k_w -= 1
    # ensure the height of the kernel is odd
    if k_h % 2 == 0:
        k_h -= 1

    try:
        blured_image = cv2.GaussianBlur(original_image, (k_w, k_h), factor)
        return blured_image
    except:
        print('Ignoring blur')
        return original_image


if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES = ['background', 'human']
    CLASS_RGB_VALUES = [[0, 0, 0], [1, 1, 1]]
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    BLUR_KERNEL = 3
    BLUR_FACTOR = 20
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)
    model = torch.load('model/best_model.pth', map_location=DEVICE)
    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        height, width, _ = image.shape
        image = cv2.resize(image, (256, 256))

        sample = preprocessing(image=image)
        preprocessed_image = torch.from_numpy(sample['image']).to(DEVICE).unsqueeze(0)

        pred_mask = model(preprocessed_image.float())
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()

        # Convert pred_mask and image from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))

        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), CLASS_RGB_VALUES)

        image = cv2.resize(image.astype(np.uint8), (width, height))
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (width, height))

        output_image = blur(image, kernel=BLUR_KERNEL, factor=BLUR_FACTOR)
        output_image[pred_mask == 1] = image[pred_mask == 1]

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Original image', cv2.flip(image, 1))
        cv2.imshow('Result', cv2.flip(output_image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
