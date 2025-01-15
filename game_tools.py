from enum import Enum
import numpy as np
import random
import string
import time
import copy
import json
from typing import List
from ml import load_model, normalize_board, cell_index_to_cell, predict_move_from_board_NN

default_state = [["-","-","-"],
                 ["-","-","-"],
                 ["-","-","-"]]

class Player(Enum):
    O ="O"
    X = "X"

def play_menu():
    print("------------------------------")
    key = input("Choose a Number :\n1. Play against Minimax\n2. Play against Neural Network\n3. Play PvP\n")
    if key == "1":
        start_pve_game_minimax()
    elif key == "2":
        start_pve_game_NN()
    elif key == "3":
        start_pvp_game()
    play_menu()

def start_pve_game_NN():
    state = copy.deepcopy(default_state)
    player = Player.O if random.random() > 0.5 else Player.X
    environment = Player.O if player.value == Player.X.value else Player.X
    # player = Player.X
    # environment = Player.O
    model_file_name = "model_o" if environment.value == Player.O.value else "model_x"
    model = load_model(model_file_name)
    render_game(state)
    for round in range(9):
        round_owner = Player.O if round % 2 == 0 else Player.X
        if round_owner.value == player.value:
            x,y = get_input(round_owner,state)
            state[x][y] = round_owner.value
        else:
            cell = predict_move_from_board_NN(model, state)
            if not is_cell_empty(state,cell):
                cell = choose_random_cell(state, round_owner)
                print("random choose! poor data")
            x,y = cell
            state[x][y] = round_owner.value
        render_game(state)
        winner = check_winner(state)
        if winner == "X" or winner == "O":
            print(f"Player {winner} has Won! Congratulations!!")
            break
    else:
        print(f"Game is a Tie!")

def start_pve_game_minimax():
    state = copy.deepcopy(default_state)
    player = Player.O if random.random() > 0.5 else Player.X
    environment = Player.O if player.value == Player.X.value else Player.X
    # player = Player.X
    # environment = Player.O
    render_game(state)
    for round in range(9):
        round_owner = Player.O if round % 2 == 0 else Player.X
        if round_owner.value == player.value:
            x,y = get_input(round_owner,state)
            state[x][y] = round_owner.value
        else:
            x,y = predict_move_from_board_minimax(state,environment)
            state[x][y] = round_owner.value
        render_game(state)
        winner = check_winner(state)
        if winner == "X" or winner == "O":
            print(f"Player {winner} has Won! Congratulations!!")
            break
    else:
        print(f"Game is a Tie!")


def start_pvp_game():
    state = copy.deepcopy(default_state) 
    render_game(state)
    for round in range(9):
        round_owner = Player.O if round % 2 == 0 else Player.X
        x,y = get_input(round_owner,state)
        state[x][y] = round_owner.value
        render_game(state)
        winner = check_winner(state)
        if winner == "X" or winner == "O":
            print(f"Player {winner} has Won! Congratulations!!")
            break
    else:
        print(f"Game is a Tie!")

def check_winner(state:list[list[str]]):
    winner = None 
    for i in range(3):
        if state[i][0] == state[i][1] == state[i][2] and state[i][0] != "-": 
            winner = state[i][0]
            break
        if state[0][i] == state[1][i] == state[2][i] and state[0][i] != "-":
            winner = state[0][i]
            break
    if state[0][0] == state[1][1] == state[2][2] and state[0][0] != "-":
        winner = state[0][0]
    if state[0][2] == state[1][1] == state[2][0] and state[0][2] != "-":
        winner = state[0][2]
    if winner == None:
        state = np.array(state).reshape(-1)
        if not "-" in state:
            winner = 0
    return winner

def check_winner_normalized(board:list[int]):
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


def render_game(state):
    print(" ", "a", "b", "c", sep="     ")
    print()
    for index, item in enumerate(state):
        print(index+1, state[index][0], state[index][1], state[index][2], sep="     ")
        print()


def get_input(player: Player, state):
    cell = []
    while True:
        chosen_cell = input(f"Player {player.value} Turn. Choose cell :")
        if len(chosen_cell) == 2 and chosen_cell[0] in {"a","b","c"} and chosen_cell[1] in {"1","2","3"}:
            cell = parse_cell(chosen_cell)
            if is_cell_empty(state,cell): break
    return cell

def is_cell_empty(board:List[List[str]], cell:List[int])-> bool:
    x,y = cell
    return board[x][y] == "-"

def parse_cell(cell_str:str):
    cell = []
    cell.append(int(cell_str[1]) -1) 
    cell.append(label_to_number(cell_str[0]))
    return cell

def label_to_number(label:str):
    number = None
    if label == 'a': number = 0
    if label == 'b': number = 1
    if label == 'c': number = 2
    return number

def number_to_label(number:int):
    label = None
    if number == 0: label = 'a'
    if number == 1: label = 'b'
    if number == 2: label = 'c'
    return label


# AI functions
def run_random_games_and_save_game_logs(count=10000, log_file_name = "game_logs"):
    game_data_list = []
    for i in range(count):
        game_data = run_random_vs_random_game()
        game_data_list.append(game_data)

    with open(f"data/{log_file_name}.json", "w") as file:
        json.dump(game_data_list, file)
    print("Games Log Saving Ended !!!")


def run_random_vs_random_game():
    game_data = {"winner": None, "rounds": [], "moves": []}
    winner = None
    state = copy.deepcopy(default_state)
    game_data["rounds"].append(state)
    for round in range(9):
        player = Player.O if round % 2 == 0 else Player.X
        x,y = choose_random_cell(state,player)
        game_data["rounds"].append(copy.deepcopy(state))
        game_data["moves"].append((x+1, y+1))
        winner = check_winner(state)
        if winner == "X" or winner == "O": break

    game_data["winner"] = winner
    return game_data


def choose_random_cell(state:List[List[str]], player:Player):
    x,y = find_random_empty_cell(state)
    state[x][y] = player.value
    return (x,y)


def find_random_empty_cell(state):
    empty_positions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == '-']
    if not empty_positions:
        raise IndexError("no more empty cell left to choose")
    return random.choice(empty_positions)

# minimax
def run_and_log_minimax_games(log_file_name:str, minimax_player:Player, count=1000, file_limit= 100):
    game_data_list = []
    for i in range(1,count):
        game_data = run_minimax_vs_random_game(minimax_player)
        game_data_list.append(game_data)
        if i%100 == 0: print(i)
        if i%file_limit == 0:
            with open(f"/mnt/d/wsl-data/minimax_{minimax_player.value}_{log_file_name}_{i/file_limit}.json", "w") as file:
                json.dump(game_data_list, file)
            game_data_list = []
    print("Games Log Saving Ended !!!")


def run_minimax_vs_random_game(minimax_player:Player):
    game_data = {"winner": None, "rounds": [], "moves": []}
    state = copy.deepcopy(default_state)
    game_data["rounds"].append(default_state)
    for round in range(9):
        round_owner = Player.O if round % 2 == 0 else Player.X
        if round_owner.value == minimax_player.value:
            x,y = predict_move_from_board_minimax(state, minimax_player)
            state[x][y] = minimax_player.value
        else:
            x,y = choose_random_cell(state, round_owner)
            state[x][y] = round_owner.value
        game_data["rounds"].append(copy.deepcopy(state))
        game_data["moves"].append((x+1, y+1))
        winner = check_winner(state)
        if winner != None:
            game_data["winner"] = winner
            break
    return game_data


def create_future_possibilities_from_board(board:list[int], player:Player):
    # board is 9 of -1,0,1
    possibilities = []
    mark = 1 if player.value == Player.O.value else -1
    for index, cell in enumerate(board):
        if board[index] == 0:
            board[index] = mark
            winner = check_winner_normalized(board)
            if(winner != None):
                possibilities.append(winner)
            else:
                p = switch_player(player)
                sub_possibilities = create_future_possibilities_from_board(board, p)
                possibilities.append(sub_possibilities)
            board[index] = 0
    return possibilities


def choose_best_cell_from_possibilities(possibilities, board:List[int], player:Player)-> int:
    reduced_p = possibilities
    while(not is_list_of_int(reduced_p)):
        reduced_p = reduce_one_round_from_possibilities(reduced_p, player,0)
    reduced_p = np.array(reduced_p)
    target = np.max(reduced_p) if player.value == Player.O.value else np.min(reduced_p)
    target_indexes = [index for index,i in enumerate(reduced_p) if i==target]
    random_idx = random.choice(target_indexes)

    empty_cell_indexes = [index for index, cell in enumerate(board) if cell == 0]
    cell_index = empty_cell_indexes[random_idx]
    cell = cell_index_to_cell(cell_index)
    return cell


def reduce_one_round_from_possibilities(possibilities, player:Player, depth:int):
    # round can be number or array of numbers or array of arrays
    if isinstance(possibilities, (int, float, np.number)): return possibilities
    elif is_list_of_int(possibilities):
        protocol = "max" if player.value == Player.O.value and depth%2 == 0 or player.value == Player.X.value and depth%2 == 1 else "min"
        target = np.max(possibilities) if protocol == 'max' else np.min(possibilities)
        return target
    else:
        return [reduce_one_round_from_possibilities(array, player, depth+1) for array in possibilities]


def is_list_of_int(array):
    return isinstance(array, list) and all(isinstance(x, (int, float, np.number)) for x in array)


def switch_player(player):
    return Player.O if player.value == Player.X.value else Player.X


def predict_move_from_board_minimax(board:List[List[str]], player:Player)-> List[int]:
   board = normalize_board(board)
   board_is_empty = not 1 in board and not -1 in board
   if board_is_empty and player.value == Player.O.value:
      cell_index = random.choice(range(9))
      cell = cell_index_to_cell(cell_index)
   else:
      possibilities = create_future_possibilities_from_board(board, player)
      cell = choose_best_cell_from_possibilities(possibilities, board, player)
   
   return cell