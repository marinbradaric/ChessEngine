import chess
import pandas as pd
import numpy as np
import gc
import re
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

NUM_POSITIONS = 50000 # Change this to specify the number of chess positions for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

chess_data = pd.read_csv("data/filtered_chess_games.csv")

def board_2_rep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep

def create_rep_layer(board, piece_type):
    s = str(board)
    s = re.sub(f"[^{piece_type}{piece_type.upper()} \n]", '.', s)
    s = re.sub(f"{piece_type}", "-1", s)
    s = re.sub(f"{piece_type.upper()}", "1", s)
    s = re.sub("\.", "0", s)

    board_mat = []
    for row in s.split("\n"):
        row = row.split(" ")
        row = [int(x) for x in row]
        board_mat.append(row)

    return np.array(board_mat)

def create_move_list(s):
    s = str(s)
    s = re.sub(r"[\[\]']", '', s)
    moves = re.sub('\d*\. ','',s).split(' ')[:-1]
    if moves[-1] == '':
        moves.pop()
    return moves

def move_2_rep(move, board):
    try:
        board.push_san(move).uci()
    except:
        return None
    move = str(board.pop())

    from_output_layer = np.zeros((8,8))
    from_row = 8 - int (move[1])
    from_column = letter_2_num [move[0]]
    from_output_layer [from_row,from_column] = 1

    to_output_layer = np.zeros((8,8))
    to_row = 8 - int(move[3])
    to_column = letter_2_num[move[2]]
    to_output_layer [to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])

class ChessDataset(Dataset):
    def __init__(self, games):
        super(ChessDataset, self).__init__()
        self.games = games

    def __len__(self):
        return NUM_POSITIONS

    def __getitem__(self, index):
        while True:
            game_i = np.random.randint(self.games.shape[0])
            random_game = chess_data['AN'].values[game_i]
            moves = create_move_list(random_game)
            if len(moves) < 2:
                continue
            game_state_i = np.random.randint(len(moves)-1)
            next_move = moves[game_state_i]
            moves = moves[:game_state_i]
            board = chess.Board()
            for move in moves:
                try:
                    board.push_san(move)
                except:
                    continue
            x = board_2_rep(board)
            y = move_2_rep(next_move, board)
            if y is None:
                continue
            if game_state_i % 2 == 1:
                x *= -1
            return x, y

data_train = ChessDataset(chess_data)
data_train_loader = DataLoader(data_train, batch_size=250, shuffle=True, drop_last=True)

class module(nn.Module):

    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x    


metric_from = nn.CrossEntropyLoss()
metric_to = nn.CrossEntropyLoss()



model = ChessNet()
model.to(device)

if __name__ == "__main__":
    # Training loop
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=0.0001)

    for epoch in range(num_epochs):
        total_loss = 0.0
        # Use tqdm to visualize progress
        with tqdm(data_train_loader, unit="batch") as t:
            for batch_idx, (x, y) in enumerate(t):
                x = x.float().to(device)
                y = y.float().to(device)

                # Forward pass
                output = model(x)

                # Calculate loss
                loss_from = metric_from(output[:, 0, :], y[:, 0, :])
                loss_to = metric_to(output[:, 1, :], y[:, 1, :])
                loss = loss_from + loss_to

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update tqdm progress bar
                t.set_postfix(loss=total_loss / (batch_idx + 1))  # Update loss in progress bar

        # Print epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_train_loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")
