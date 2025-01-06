from enum import Enum
import numpy as np
import json
import random
from typing import List
from dataclasses import dataclass
from tensorflow.keras import models, layers

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
   # won_game_logs = [log for log in game_logs if log["winner"] == player.value]
   not_lost_game_logs = [log for log in game_logs if log["winner"] == player.value or log["winner"] == 0]
   start_round = 0 if player.value == Player.O.value else 1
   scenarios = []
   for game_log in not_lost_game_logs:
      rounds_count_without_result_round = len(game_log["rounds"]) -1
      for round_idx in range(start_round, rounds_count_without_result_round, 2):
         board = normalize_board(game_log["rounds"][round_idx])
         x,y = np.array(game_log["moves"][round_idx])
         cell_chances = np.zeros((3, 3))
         cell_chances[x-1][y-1] = 1
         cell_chances = cell_chances.reshape(9)
         scenario = Scenario(board=board, cell_chance=cell_chances)
         scenarios.append(scenario)
   dataset = Dataset(player=normalize_shape(player.value), scenarios=scenarios)
   with open(f"data/{logs_file_name}_dataset_player_{player.value}.json", "w") as file:
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

def normalize_board(board)-> List[int]:
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
   model = load_model(model_file_name)
   model.fit(train_boards, train_moves, epochs=10, batch_size=1, validation_split=0.2)
   test_accuracy = model.evaluate(test_boards, test_moves)
   print(f"Test accuracy: {test_accuracy}")
   model.save(f"models/{dataset_file_name}_model.keras")


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

   test_accuracy = model.evaluate(test_boards, test_moves)
   print(f"Test accuracy: {test_accuracy}")
   model.save(f"models/{dataset_file_name}_model.keras")

def predict_move_from_board_NN(model, board:List[List[str]])-> List[int]:
   board = normalize_board(board)
   board = board.reshape(1,-1)
   cells_chance = model.predict(board)
   print(cells_chance)
   move_idx = np.argmax(cells_chance)
   move_cell = cell_index_to_cell(move_idx)
   return move_cell

def load_model(model_file_name:str):
   model = models.load_model(f'models/{model_file_name}.keras')
   return model

def cell_index_to_cell(index):
   return (index // 3, index % 3)
