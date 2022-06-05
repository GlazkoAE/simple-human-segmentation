import cv2
import torch
import numpy as np

from dataset import colour_code_segmentation, reverse_one_hot

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES = ['background', 'human']
    CLASS_RGB_VALUES = [[0, 0, 0], [1, 1, 1]]
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
        tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2)
        pred_mask = model(tensor.float())
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()

        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))

        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), CLASS_RGB_VALUES)

        image = cv2.resize(image.astype(np.uint8), (width, height))
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (width, height))
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Original image', cv2.flip(image, 1))
        cv2.imshow('Predicted mask', cv2.flip(pred_mask, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
