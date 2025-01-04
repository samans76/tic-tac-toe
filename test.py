import numpy as np
from enum import Enum

# last_members = np.random.choice([1, -1, 0], size=(4, 3, 3,2))
tree = [[
  [[ 1 , 1], [-1 , 1], [ 1, -1]],
  [[ 0, -1], [ 1,  0], [ 1 , 0]],
  [[ 0 , 0], [-1,  0], [ 1 , 1]]],

 [[1,         0,        [ 1,  1]],
  [[-1,  0],    [ 1 , 1],   [ 0 , 1]],
  [-1,        [ 1 , 0],   [ 0 , 1]]],

 [[[-1, -1],   [ 0 , 0],   -1],
  [[-1 , 0],   [ 0 , 0],   [ 1 , 0]],
  [[ 0 , 0],   0,        [ 0 , 1]]],

 [[[ 1 , 1],   1,        [ 1 , 1]],
  [[-1 , 0],   [ 1 ,-1],   [ 1 ,-1]],
  [[ 0 ,-1],   [ 0 , 1],   0]]]
print(tree.shape)

# get the index that leads to win
[i for i in tree]

class Player(Enum):
    O ="O"
    X = "X"


def choose_from_layer(layer, player:Player, layer_depth:int):
    # layer can be number or array of numbers or array of arrays
    if isinstance(layer, (int, float, np.number)): return layer
    elif isinstance(layer, list[int]):
        protocol = "max" if player == Player.O and layer_depth%2 == 0 or player == Player.X and layer_depth%2 == 1 else "min"
        target = np.max(layer) if protocol == 'max' else np.min(layer)
        return target
    else:
        return [choose_from_layer(array, player, layer_depth+1) for array in layer]



def choose_from_tree(tree)-> int:
    reduced_tree = tree
    while(not isinstance(reduced_tree, list[int])):
        reduced_tree = choose_from_layer(reduced_tree)
    # return a random index of max from reduced_tree
    
# works left :
    # check if this code finds a a random index of max from reduced tree
    # write a code to create the tree
