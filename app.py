from game_tools import start_pvp_game,start_pve_game_minimax,start_pve_game_NN, run_random_games_and_save_game_logs, run_and_log_minimax_games, Player
from ml import create_dataset_from_game_logs, train_model, load_model, predict_move_from_board_NN,continue_model_train

import os
os.makedirs("data", exist_ok=True)

# # Create Game Logs
# run_random_games_and_save_game_logs(500000, "logs_500k_6")

## Save minimax vs random game logs
# run_and_log_minimax_games("logs_1.2k", Player.X, 1200)


# Create Dataset
# create_dataset_from_game_logs("minimax_X_logs_0.5k", Player.X)

## Train Model
# train_model("minimax_X_logs_0.5k_dataset_player_X")

# Continue Model Train
# continue_model_train("minimax_X_logs_1.1k_dataset_player_X_model","minimax_X_logs_1.2k_dataset_player_X")


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
#     start_pvp_game()


## Play PvE with minimax
# while True:
#     input("Press Any Key To Start The Game!")
    # start_pve_game_minimax()


## Play PvE with Neural Network
# while True:
#     input("Press Any Key To Start The Game!")
#     start_pve_game_NN()
    