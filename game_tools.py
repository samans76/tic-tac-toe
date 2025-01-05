from enum import Enum
import numpy as np
import random
import copy
import json
from typing import Tuple
from ml import load_model, predict_move_from_board_NN,predict_move_from_board_minimax

default_state = [["-","-","-"],
                 ["-","-","-"],
                 ["-","-","-"]]

class Player(Enum):
    O ="O"
    X = "X"

def start_pve_game_NN():
    state = copy.deepcopy(default_state)
    # player = Player.O if random.random() >0.5 else Player.X
    # environment = Player.O if player == Player.X else Player.X
    player = Player.O
    environment = Player.X
    model_file_name = "model_o" if environment == Player.O else "model_x"
    model = load_model(model_file_name)
    render_game(state)
    for round in range(9):
        round_owner = Player.O if round % 2 == 0 else Player.X
        if round_owner == player:
            x,y = get_input(round_owner,state)
            state[x][y] = round_owner.name
        else:
            cell = predict_move_from_board_NN(model, state)
            if not is_cell_empty(state,cell):
                cell = choose_cell(state, round_owner)
                print("random choose! poor data")
            x,y = cell
            state[x][y] = round_owner.name
        render_game(state)
        winner = check_winner(state)
        if winner == "X" or winner == "O":
            print(f"Player {winner} has Won! Congratulations!!")
            break
    else:
        print(f"Game is a Tie!")

def start_pve_game_minimax():
    state = copy.deepcopy(default_state)
    # player = Player.O if random.random() >0.5 else Player.X
    # environment = Player.O if player == Player.X else Player.X
    player = Player.X
    environment = Player.O
    render_game(state)
    for round in range(9):
        round_owner = Player.O if round % 2 == 0 else Player.X
        if round_owner == player:
            x,y = get_input(round_owner,state)
            state[x][y] = round_owner.name
        else:
            cell = []
            while(True):
                cell = predict_move_from_board_minimax(state,environment)
                print("cell :",cell)
                if is_cell_empty(state,cell): break
            x,y = cell
            state[x][y] = round_owner.name
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
        state[x][y] = round_owner.name
        render_game(state)
        winner = check_winner(state)
        if winner == "X" or winner == "O":
            print(f"Player {winner} has Won! Congratulations!!")
            break
    else:
        print(f"Game is a Tie!")

def check_winner(state):
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

    
def render_game(state):
    print(" ", "a", "b", "c", sep="     ")
    print()
    for index, item in enumerate(state):
        print(index+1, state[index][0], state[index][1], state[index][2], sep="     ")
        print()


def get_input(player: Player, state):
    cell = []
    while True:
        chosen_cell = input(f"Player {player.name} Turn. Choose cell :")
        if len(chosen_cell) == 2 and chosen_cell[0] in {"a","b","c"} and chosen_cell[1] in {"1","2","3"}:
            cell = parse_cell(chosen_cell)
            if is_cell_empty(state,cell): break
    return cell

def is_cell_empty(board:list[list[str]], cell:Tuple[int, int])-> bool:
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
def choose_cell(state, player:Player):
    x,y = find_random_empty_cell(state)
    state[x][y] = player.name
    return (x,y)

def find_random_empty_cell(state):
    empty_positions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == '-']
    if not empty_positions:
        raise IndexError("no more empty cell left to choose")
    return random.choice(empty_positions)

def run_virtual_game():
    game_data = {"winner": None, "rounds": [], "moves": []}
    winner = None
    state = copy.deepcopy(default_state)
    game_data["rounds"].append(state)
    for round in range(9):
        player = Player.O if round % 2 == 0 else Player.X
        cell = choose_cell(state,player)
        game_data["rounds"].append(copy.deepcopy(state))
        game_data["moves"].append((cell[0]+1, cell[1]+1))
        winner = check_winner(state)
        if winner == "X" or winner == "O": break

    game_data["winner"] = winner
    return game_data

def run_random_games_and_save_game_logs(count=10000, log_file_name = "game_logs"):
    game_data_list = []
    for i in range(count):
        game_data = run_virtual_game()
        game_data_list.append(game_data)

    with open(f"data/{log_file_name}.json", "w") as file:
        json.dump(game_data_list, file)
    print("Games Log Saving Ended !!!")

def run_minimax_game(player:Player):
    game_data = {"winner": None, "rounds": [], "moves": []}
    state = copy.deepcopy(default_state)
    game_data["rounds"].append(default_state)
    for round in range(9):
        round_owner = Player.O if round % 2 == 0 else Player.X
        if round_owner.name == player.name:
            x,y = predict_move_from_board_minimax(state,player)
            state[x][y] = player.name
        else:
            x,y = choose_cell(state, round_owner)
            state[x][y] = round_owner.name
        game_data["rounds"].append(copy.deepcopy(state))
        game_data["moves"].append((x+1, y+1))
        winner = check_winner(state)
        if winner != None:
            game_data["winner"] = winner
            break
    return game_data

def run_and_log_minimax_games(log_file_name:str, player:Player,count=1000):
    game_data_list = []
    for i in range(count):
        if i%10 == 0: print(i)
        game_data = run_minimax_game(player)
        game_data_list.append(game_data)

    with open(f"data/minimax_{player.name}_{log_file_name}.json", "w") as file:
        json.dump(game_data_list, file)
    print("Games Log Saving Ended !!!")