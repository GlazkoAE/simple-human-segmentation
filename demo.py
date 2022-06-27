import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from dataset import colour_code_segmentation, reverse_one_hot, get_preprocessing

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES = ['background', 'human']
    CLASS_RGB_VALUES = [[0, 0, 0], [1, 1, 1]]
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
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

        image.flags.writeable = False
        height, width = image.shape[:2]

        sample = preprocessing(image=cv2.resize(image, (256, 256)))
        preprocessed_image = torch.from_numpy(sample['image']).to(DEVICE).unsqueeze(0)

        pred_mask = model(preprocessed_image.float())
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()

        # Convert pred_mask and image from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))

        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), CLASS_RGB_VALUES)

        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (width, height))

        # output_image = blur(image, kernel=BLUR_KERNEL, factor=BLUR_FACTOR)
        output_image = cv2.GaussianBlur(image, (-1, -1), BLUR_FACTOR)
        output_image[pred_mask == 1] = image[pred_mask == 1]

        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('Original image', cv2.flip(image, 1))
        cv2.imshow('Background blur', cv2.flip(output_image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
