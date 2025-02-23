from model import ClassifierModel

import torch
import torch.nn as nn

import numpy as np
import os
import cv2
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Animal Classifier Video - Inference Script")
    parser.add_argument('--video-path', '-p', type=str, required=True, help='Path to the input video.')
    parser.add_argument('--frame-size', '-s', type=int, default=224, help='Frame size for resizing (default: 224).')
    parser.add_argument('--checkpoint-dir', '-c', type=str, default='./checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--checkpoint-name', '-n', type=str, default='best_model.pt', help='Checkpoint file name (default: best_model.pt).')
    parser.add_argument('--output-dir', '-o', type=str, default='./outputs', help='Directory to save the output video (default: ./outputs).')
    parser.add_argument('--show-video', action='store_true', help='Display the video with predictions in real-time.')
    return parser.parse_args()


def load_model(checkpoint_dir, checkpoint_name, device):
    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), map_location=device)
    model = ClassifierModel(num_classes=len(checkpoint['classes']))
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    return model, checkpoint['classes']


def preprocess_frame(frame, frame_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (frame_size, frame_size))
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))[None, :, :, :]
    return torch.tensor(frame, dtype=torch.float32)


def process_video(video_path, model, classes, frame_size, output_path, device, show_video=False):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    # Set up video writer
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_tensor = preprocess_frame(frame, frame_size).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prob = nn.functional.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(prob, dim=0)
            predicted_class = classes[predicted_idx.item()]

        # Annotate the frame
        label = f"{predicted_class} ({confidence.item() * 100:.2f}%)"
        cv2.putText(frame, label, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10, cv2.LINE_AA)
        output_video.write(frame)

        # Show the frame if requested
        if show_video:
            cv2.imshow("Prediction", cv2.resize(frame, (800, 400)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    output_video.release()
    if show_video:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, classes = load_model(args.checkpoint_dir, args.checkpoint_name, device)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}_output.mp4")
    process_video(args.video_path, model, classes, args.frame_size, output_path, device, args.show_video)
    print(f"Processed video saved to {output_path}")
