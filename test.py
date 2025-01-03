import numpy as np
from enum import Enum

# last_members = np.random.choice([1, -1, 0], size=(4, 3, 3,2))
tree = np.array(
[[
  [[ 1 , 1], [-1 , 1], [ 1, -1]],
  [[ 0, -1], [ 1,  0], [ 1 , 0]],
  [[ 0 , 0], [-1,  0], [ 1 , 1]]],

 [[1,           0,          [ 1,  1]],
  [[-1,  0],    [ 1 , 1],   [ 0 , 1]],
  [-1,          [ 1 , 0],   [ 0 , 1]]],

 [[[-1, -1],   [ 0 , 0],   -1],
  [[-1 , 0],   [ 0 , 0],   [ 1 , 0]],
  [[ 0 , 0],   0,          [ 0 , 1]]],

 [[[ 1 , 1],   1,          [ 1 , 1]],
  [[-1 , 0],   [ 1 ,-1],   [ 1 ,-1]],
  [[ 0 ,-1],   [ 0 , 1],   0]]])
print(last_members)

class Player(Enum):
    O ="O"
    X = "X"


def choose_from_layer(layer, player:Player, layer_depth:int)-> np.number:
    # layer can be array or number
    if isinstance(layer, (int, float, np.number)): return layer
    elif isinstance(layer, np.ndarray):
        protocol = "max" if player == Player.O and layer_depth%2 == 0 or player == Player.X and layer_depth%2 == 1 else "min"
        target = np.max(layer) if protocol == 'max' else np.min(layer)
        return target

