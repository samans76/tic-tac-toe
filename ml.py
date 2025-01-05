from enum import Enum
import numpy as np
import json
import random
from typing import List
from dataclasses import dataclass
from tensorflow.keras import models, layers, regularizers
# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

class Player(Enum):
    O ="O"
    X = "X"

@dataclass
class Scenario:
   board: np.ndarray
   cell_chance: np.ndarray

   def to_serializable(self):
     return {"board": self.board.tolist(), "cells_chance": self.cell_chance.tolist()} 

@dataclass
class Dataset:
   player: int
   scenarios: List[Scenario]

   def to_serializable(self):
      return {"player": self.player, "scenarios": [i.to_serializable() for i in self.scenarios]} 


def create_dataset_from_game_logs(logs_file_name:str, player:Player):
   with open(f"data/{logs_file_name}.json", 'r') as file: 
      game_logs = json.load(file)
   # won_game_logs = [log for log in game_logs if log["winner"] == player.name]
   not_lost_game_logs = [log for log in game_logs if log["winner"] == player.name or log["winner"] == 0]
   start_round = 0 if player == Player.O else 1
   scenarios = []
   for game_log in not_lost_game_logs:
      rounds_count_without_result_round = len(game_log["rounds"]) -1
      for round_idx in range(start_round, rounds_count_without_result_round, 2):
         board = normalize_board(game_log["rounds"][round_idx])
         x,y = np.array(game_log["moves"][round_idx])
         cells = np.zeros((3, 3))
         cells[x-1][y-1] = 1
         cells = cells.reshape(9)
         scenario = Scenario(board=board, cell_chance=cells)
         scenarios.append(scenario)
   dataset = Dataset(player=normalize_shape(player.name), scenarios=scenarios)
   with open(f"data/{logs_file_name}_dataset_player_{player.name}.json", "w") as file:
      json.dump(dataset.to_serializable(), file)
   print("Dataset Saving Ended !!!")
   return dataset

def normalize_moves(moves):
   # (<9,2) int -> (18)
   moves = np.array(moves).reshape(-1)
   moves = moves / 3
   moves_length = len(moves)
   if moves_length < 18:
      moves = np.append(moves, [0] * (18 - moves_length))
   return moves

def normalize_board(board):
   # (3,3) string -> (9) int
   board = np.array(board).reshape(-1)
   board = np.array([normalize_shape(i) for i in board])
   return board

def normalize_shape(shape):
   if shape == 'X': return -1
   if shape == 'O': return 1
   else:  return 0

def continue_model_train(model_file_name:str,dataset_file_name ):
   with open(f"data/{dataset_file_name}.json", 'r') as file: 
      dataset = json.load(file)
   scenarios = np.array(dataset["scenarios"])
   mask = np.array([random.random() < 0.8 for i in scenarios])
   train_scenarios = scenarios[mask]
   test_scenarios = scenarios[~mask]
   train_boards = np.array([scenario["board"] for scenario in train_scenarios]) # (-1, 9)
   train_moves = np.array([scenario["cells_chance"] for scenario in train_scenarios]) # (-1, 9)
   test_boards = np.array([scenario["board"] for scenario in test_scenarios]) 
   test_moves = np.array([scenario["cells_chance"] for scenario in test_scenarios]) 
   # model = load_model(model_file_name)
   model = models.load_model(f'data/{model_file_name}.keras')
   model.fit(train_boards, train_moves, epochs=10, batch_size=1, validation_split=0.2)
   test_accuracy = model.evaluate(test_boards, test_moves)
   print(f"Test accuracy: {test_accuracy}")
   model.save(f"data/{dataset_file_name}_model.keras")


def train_model(dataset_file_name):
   with open(f"data/{dataset_file_name}.json", 'r') as file: 
      dataset = json.load(file)
   scenarios = np.array(dataset["scenarios"])
   mask = np.array([random.random() < 0.8 for i in scenarios])
   train_scenarios = scenarios[mask]
   test_scenarios = scenarios[~mask]
   train_boards = np.array([scenario["board"] for scenario in train_scenarios]) # (-1, 9)
   train_moves = np.array([scenario["cells_chance"] for scenario in train_scenarios]) # (-1, 9)
   test_boards = np.array([scenario["board"] for scenario in test_scenarios]) 
   test_moves = np.array([scenario["cells_chance"] for scenario in test_scenarios]) 
   model = models.Sequential([
      layers.Dense(400, activation='tanh', input_shape=(9,)),
      layers.Dropout(0.2),                                         # Dropout for regularization
      layers.Dense(300, activation='tanh'),                       # Hidden layer
      layers.Dropout(0.2),                                         # Dropout for regularization
      layers.Dense(200, activation='tanh'),                       # Hidden layer
      layers.Dense(9, activation='softmax')                      # Output layer (9 classes)
   ])
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   model.fit(train_boards, train_moves, epochs=10, batch_size=1, validation_split=0.2)
   # model = models.Sequential([
   #    layers.Reshape((3, 3, 1), input_shape=(9,)),  # Reshape input to 3x3 grid
   #    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # 2D Convolution to capture spatial patterns
   #    layers.MaxPooling2D((2, 2)),  # Max-pooling to reduce spatial dimensions
   #    layers.Flatten(),  # Flatten the output from 2D to 1D
   #    layers.Dense(64, activation='relu'),  # Fully connected layer
   #    layers.Dropout(0.2),  # Dropout for regularization
   #    layers.Dense(9, activation='softmax')  # Output layer with softmax for 9 possible moves
   # ])
   # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   # model.fit(train_boards, train_moves, epochs=10, batch_size=64, validation_split=0.2)

   test_accuracy = model.evaluate(test_boards, test_moves)
   print(f"Test accuracy: {test_accuracy}")
   model.save(f"data/{dataset_file_name}_model.keras")

def predict_move_from_board_NN(model, board:List[List[str]])-> List[int]:
   board = normalize_board(board)
   board = board.reshape(1,-1)
   print("board : ", board)
   cells_chance = model.predict(board)
   print("cells_chance : ", cells_chance)
   move_idx = np.argmax(cells_chance)
   move_cell = cell_index_to_cell(move_idx)
   return move_cell

def load_model(model_file_name:str):
   model = models.load_model(f'data/{model_file_name}.keras')
   return model

def cell_index_to_cell(index):
   return (index // 3, index % 3)

# predict
# board = [["O","-","-"],
#          ["O","X","-"],
#          ["-","-","X"]]
# model = load_model("logs_10000_dataset_player_O_model")
# move = predict_move_from_board(model,board)
# print("last move :",move)

def choose_best_from_possibilities(possibilities, player:Player)-> int:
    reduced_p = possibilities
    while(not is_list_of_int(reduced_p)):
        reduced_p = choose_from_layer(reduced_p, player,0)
    reduced_p = np.array(reduced_p)
    target = np.max(reduced_p) if player.name == Player.O.name else np.min(reduced_p)
    target_indexes = [index for index,i in enumerate(reduced_p) if i==target]
    random_idx = random.choice(target_indexes)
    return random_idx

def choose_from_layer(layer, player:Player, layer_depth:int):
    # layer can be number or array of numbers or array of arrays
    if isinstance(layer, (int, float, np.number)): return layer
    elif is_list_of_int(layer):
        protocol = "max" if player.name == Player.O.name and layer_depth%2 == 0 or player.name == Player.X.name and layer_depth%2 == 1 else "min"
        target = np.max(layer) if protocol == 'max' else np.min(layer)
        return target
    else:
        return [choose_from_layer(array, player, layer_depth+1) for array in layer]

def is_list_of_int(array):
    return isinstance(array, list) and all(isinstance(x, (int, float, np.number)) for x in array)

def create_possibilities_from_board(board, player:Player):
    # board is 9 of -1,0,1
    possibilities = []
    mark = 1 if player.name == Player.O.name else -1
    for index, cell in enumerate(board):
        if board[index] == 0:
            board[index] = mark
            winner = check_winner_normalized(board)
            if(winner != None):
                possibilities.append(winner)
            else:
                p = switch_player(player)
                sub_possibilities = create_possibilities_from_board(board, p)
                possibilities.append(sub_possibilities)
            board[index] = 0
    return possibilities

def switch_player(player):
    return Player.O if player.name == Player.X.name else Player.X

def check_winner_normalized(board):
   # board is 9 of -1,0,1
   
    board = np.array(board).reshape(3,3)
    winner = None 
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != 0: 
            winner = board[i][0]
            break
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != 0:
            winner = board[0][i]
            break
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
        winner = board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
        winner = board[0][2]
    if winner == None and not 0 in board:
        winner = 0
    return winner

def predict_move_from_board_minimax(board:List[List[str]], player:Player)-> List[int]:
   board = normalize_board(board)
   possibilities = create_possibilities_from_board(board, player)
   possibility_idx = choose_best_from_possibilities(possibilities, player)
   empty_cell_indexes = [index for index, cell in enumerate(board) if cell == 0]
   cell = cell_index_to_cell(empty_cell_indexes[possibility_idx])
   return cell