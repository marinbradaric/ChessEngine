import torch
from main import ChessNet, board_2_rep, letter_2_num, num_2_letter, create_rep_layer
import chess
import numpy as np

MODEL_PATH = "models/model.pth" # Specify the path to the pre-trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device ", device)

model = ChessNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

def predict(board_rep):
    model.eval()
    with torch.no_grad():
        output = model(board_rep)
        return output

def check_mate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return True
        _ = board.pop()


def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs ** 3
    probs = probs / probs.sum()
    return probs

def choose_move(board, color):
    legal_moves = list(board.legal_moves)

    move = check_mate_single(board)
    if move is not None:
        return move
    
    x = torch.Tensor(board_2_rep(board)).float().to(device)
    if color == chess.BLACK:
        x = x * -1
    x = x.unsqueeze(0)
    move = predict(x)

    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))
    for from_ in froms:
        val = move[0, 0, 8 - int(from_[1]), letter_2_num[from_[0]]]
        vals.append(val)

    probs = distribution_over_moves(vals)

    choosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        if from_ == choosen_from:
            to = str(legal_move)[2:]
            val = move[0, 1, 8 - int(to[1]), letter_2_num[to[0]]]
            vals.append(val)
        else:
            vals.append(0)

    choosen_move = legal_moves[np.argmax(vals)]
    return choosen_move

def play():
    while True:
        fen_position = input("Enter the FEN position (or type 'exit' to quit): ")
        if fen_position.lower() == "exit":
            print("Exiting...")
            break

        board = chess.Board(fen_position)
        if board.turn == chess.WHITE:
            color = chess.WHITE
        else:
            color = chess.BLACK

        move = choose_move(board, color)
        print("Best move:", move)
        print()

play()
