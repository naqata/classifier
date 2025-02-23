from model import ClassifierModel

import torch
import torch.nn as nn

import numpy as np
import os
import cv2
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Animal Classifier - Inference Script")
    parser.add_argument('--image-path', '-p', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--image-size', '-s', type=int, default=224, help='Size to resize the input image (default: 224).')
    parser.add_argument('--checkpoint-dir', '-c', type=str, default='./checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--checkpoint-name', '-n', type=str, default='best_model.pt', help='Name of the checkpoint file (default: best_model.pt).')
    return parser.parse_args()

def preprocess_image(image_path, image_size):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f':::Image not found at path: {image_path}')
    
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    
    return torch.tensor(image, dtype=torch.float32), original_image

def load_model(checkpoint_dir, checkpoint_name, device):
    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), map_location=device)
    classes = checkpoint['classes']
    model = ClassifierModel(num_classes=len(classes))
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    return model, classes    


def predict(model, image, classes, device):
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        prob = nn.functional.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(prob, dim=0)
        predicted_class = classes[predicted_idx]

    return predicted_class, confidence


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, classes = load_model(args.checkpoint_dir, args.checkpoint_name, device)
    image_tensor, original_image = preprocess_image(args.image_path, args.image_size)
    predicted_class, confidence = predict(model, image_tensor, classes, device)

    cv2.imshow(f'Prediction: {predicted_class} ({(confidence * 100):.2f}%)', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()