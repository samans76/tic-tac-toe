from game_tools import start_pvp_game,start_pve_game, run_random_games_and_save_game_logs, Player
from ml import create_dataset_from_game_logs, train_model, load_model, predict_move_from_board

import os
os.makedirs("data", exist_ok=True)

# # Create Game Logs
run_random_games_and_save_game_logs(5000000, "logs_5m")


# Create Dataset
# create_dataset_from_game_logs("logs_200k", Player.X)


## Train Model
# train_model("logs_200k_dataset_player_X")


## Predict
# ["X", "X", "O","X", "O", "O","-", "O", "-"]
# board = [["X", "X", "O"],
#          ["X", "O", "O"],
#          ["-", "O", "-"]]
# model = load_model("logs_50k_dataset_player_X_model")
# move = predict_move_from_board(model,board)
# print("last move :",move)


## PvP Play 
# while True:
#     input("Press Any Key To Start The Game!")
#     start_game()


## Play PvE 
# while True:
#     input("Press Any Key To Start The Game!")
#     start_pve_game()