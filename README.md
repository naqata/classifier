# Animal Classifier

This repository contains scripts for training and performing inference using an animal classification model. The model is trained on an animal dataset and can classify images and videos.

## Project Structure
```
📂 Animal-Classifier
│── 📂 checkpoints/        # Directory for saving model checkpoints
│── 📂 logs/               # TensorBoard logs
│── 📂 outputs/            # Directory for saving processed videos
│── 📂 dataset/            # Dataset directory (if applicable)
│── 📄 animals_train.py    # Script for training the model
│── 📄 animals_inference_image.py  # Script for image classification inference
│── 📄 animals_inference_video.py  # Script for video classification inference
│── 📄 animals_dataset.py  # Dataset class
│── 📄 animals_model.py    # Model definition
│── 📄 requirements.txt    # Python dependencies
│── 📄 README.md           # Project documentation
```

## Installation
### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- TensorBoard

### Install Dependencies
Run the following command to install required libraries:
```bash
pip install -r requirements.txt
```

## Training the Model
To train the model, use:
```bash
python animal_train.py --data-path ./dataset --epochs 50 --batch-size 64 --learning-rate 0.001
```
**Arguments:**
- `--data-path`: Path to the dataset
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate
- `--resume`: Resume training from the last checkpoint (optional)

## Tensorboard logs
```bash
tensorboard --logdir ./logs
```

## Image Inference
To classify a single image:
```bash
python animals_inference_image.py --image-path ./test_image.jpg --checkpoint-dir ./checkpoints
```
**Arguments:**
- `--image-path`: Path to the input image
- `--checkpoint-dir`: Directory containing model checkpoints
- `--checkpoint-name`: Name of the checkpoint file (default: best_model.pt)

## Video Inference
To classify animals in a video:
```bash
python animals_inference_video.py --video-path ./test_video.mp4 --checkpoint-dir ./checkpoints --show-video
```
**Arguments:**
- `--video-path`: Path to the input video
- `--frame-size`: Size to resize frames (default: 224)
- `--checkpoint-dir`: Directory containing model checkpoints
- `--checkpoint-name`: Name of the checkpoint file (default: best_model.pt)
- `--show-video`: Display the video with predictions in real-time

## Results & Outputs
- Training logs are stored in `logs/` (can be viewed using TensorBoard).
- Checkpoints are saved in `checkpoints/`.
- Processed videos are saved in `outputs/`.

## Contributing
