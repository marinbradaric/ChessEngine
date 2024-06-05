# Chess Engine - A Convolutional Neural Network for Chess

## Project Description

This project is a Chess Engine developed using a Convolutional Neural Network (CNN) in PyTorch. The engine is trained to predict the best move from a given board position. It analyzes random positions from a dataset of high-rated chess games and learns to play chess by training on these positions.

## Installation

To set up and run the project, you'll need to install the following Python packages:

- pandas
- numpy
- chess
- tqdm
- torch (PyTorch)

You can install these packages using pip:

`pip install pandas numpy chess tqdm torch`

## Usage

### Training the Model

The training script is located in main.py. The ChessDataset class loads and processes the chess games dataset, while the ChessNet class defines the neural network architecture.

To start training, run the following command:

`python main.py`

### Predicting the Best Move

The prediction script is located in predict.py. This script loads a pre-trained model and takes a FEN (Forsyth-Edwards Notation) position as input, returning the best move according to the model.

To predict the best move for a given position, run the following command and follow the prompts:

`python predict.py`

### Customizing Training and Prediction

You can customize the training process and prediction by modifying certain parameters in the code:

- **Training**: In `main.py`, you can change the `NUM_POSITIONS` variable to specify the number of chess positions used for training. By default, it's set to 50,000.

- **Prediction**: In `predict.py`, you can specify the path to the pre-trained model by modifying the `MODEL_PATH` variable. By default, it's set to model2.pth

## Features

- **Convolutional Neural Network (CNN)**: Utilizes a deep CNN with multiple hidden layers for training and prediction.
- **High-Rated Games Dataset**: Trains on a dataset of high-rated chess games.
- **Flexible Prediction**: Provides a script for predicting the best move from any given board position using a trained model.

## Project Structure

The project directory is organized as follows:

`ChessEngine/  
├── data/  
│   └── filtered_chess_games.csv  
├── models/  
│   └── model.pth  
├── main.py  
├── predict.py  
└── README.md
└── LICENCE`

- data/: Contains the dataset of filtered high-rated chess games.
- models/: Contains pre-trained models.

  - model.pth: Trained on an unspecified number of chess positions.

- main.py: Script for training the model.
- predict.py: Script for predicting the best move from a given FEN position.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
