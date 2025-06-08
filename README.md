# Rice Variety Classification

This project implements a deep learning model to classify different varieties of rice using computer vision. The model is trained on a dataset of rice images and can identify five different varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.

## Features

- CNN-based image classification model
- Streamlit web interface for easy interaction
- Docker support for containerization
- Training visualization and metrics
- Support for multiple rice varieties

## Dataset

The dataset used for this project is sourced from Kaggle. You can find it [here](https://www.kaggle.com/datasets/your-kaggle-dataset-link).

## Project Structure

```
.
├── app.py              # Streamlit web application
├── train.py           # Model training script
├── requirements.txt   # Python dependencies
├── Dockerfile        # Docker configuration
├── model.h5          # Trained model (generated after training)
├── classes.json      # Class labels (generated after training)
└── rice_dataset/     # Dataset directory (not included in the repository)
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rice-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Organize your rice images into the appropriate subdirectories under `rice_dataset/`
   - Each variety should have its own folder with corresponding images

4. Train the model:
```bash
python train.py --data_dir ./rice_dataset --epochs 10 --batch_size 32 --img_size 128
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t rice-sorter .
```

2. Run the container:
```bash
docker run -p 3001:80 rice-sorter
```

The application will be available at `http://localhost:3001`

## Model Architecture

The model uses a CNN architecture with:
- 4 Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers with dropout for regularization
- Softmax output layer for classification

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
