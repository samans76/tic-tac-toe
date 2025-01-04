import numpy as np
from enum import Enum
import random

# last_members = np.random.choice([1, -1, 0], size=(4, 3, 3,2))
tree = [[
  [[ 1 , 1], [-1 , 1], [ 1, -1]],
  [[ 0, -1], [ 1,  0], [ 1 , 0]],
  [[ 0 , 0], [-1,  0], [ 1 , 1]]],
[1,-2],
[[2,1],[-1,-2]],
 [[1,         0,        [ 1,  1]],
  [[-1,  0],    [ 1 , 1],   [ 0 , 1]],
  [-1,        [ 1 , 0],   [ 0 , 1]]],

 [[[-1, -1],   [ 0 , 0],   -1],
  [[-1 , 0],   [ 0 , 0],   [ 1 , 0]],
  [[ 0 , 0],   0,        [ 0 , 1]]],

 [[[ 1 , 1],   1,        [ 1 , 1]],
  [[-1 , 0],   [ 1 ,-1],   [ 1 ,-1]],
  [[ 0 ,-1],   [ 0 , 1],   0]]]

class Player(Enum):
    O ="O"
    X = "X"


def choose_from_layer(layer, player:Player, layer_depth:int):
    # layer can be number or array of numbers or array of arrays
    if isinstance(layer, (int, float, np.number)): return layer
    elif is_list_of_int(layer):
        protocol = "max" if player == Player.O and layer_depth%2 == 0 or player == Player.X and layer_depth%2 == 1 else "min"
        target = np.max(layer) if protocol == 'max' else np.min(layer)
        return target
    else:
        return [choose_from_layer(array, player, layer_depth+1) for array in layer]

def choose_from_tree(tree, player:Player)-> int:
    reduced_tree = tree
    while(not is_list_of_int(reduced_tree)):
        reduced_tree = choose_from_layer(reduced_tree, player,0)
    reduced_tree = np.array(reduced_tree)
    max = np.max(reduced_tree)
    max_indexes = [index for index,i in enumerate(reduced_tree) if i==max]
    random_idx = random.choice(max_indexes)
    return random_idx

def is_list_of_int(array):
    return isinstance(array, list) and all(isinstance(x, (int, float, np.number)) for x in array)

def create_possibilities_from_board(board, player:Player):
    # board is 9 of -1,0,1
    possibilities = []
    mark = 1 if player == Player.O else -1
    for index, cell in enumerate(board):
        if board[index] == 0:
            board[index] = mark
            winner = check_winner(board)
            if(winner != None):
                possibilities.append(winner)
            else:
                p = switch_player(player)
                sub_possibilities = create_possibilities_from_board(board, p)
                possibilities.append(sub_possibilities)
            board[index] = 0
    return possibilities

def switch_player(player):
    return Player.O if player == Player.X else Player.X

def check_winner(board):
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


# print(idx)

board = [1,-1,0,-1,-1,1,1,0,0]
possibilities = create_possibilities_from_board(board, Player.O)
idx = choose_from_tree(possibilities, Player.O)
print(idx)

# test_board = [-1,1,-1,1,-1,-1,1,-1,1]
# a = check_winner(test_board)
# print(a)