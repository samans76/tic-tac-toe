from game_tools import play_menu, run_and_log_minimax_games
from ml import create_dataset_from_game_logs, train_model, load_model, continue_model_train
import os
os.makedirs("data", exist_ok=True)

play_menu()
    

## Train new NN models

# 1. Create Game Logs
# run_and_log_minimax_games("logs_8k", Player.O, 8000)

# 2. Create Dataset
# create_dataset_from_game_logs("minimax_O_logs_8k", Player.O)

# 3. Train Model
# train_model("minimax_O_logs_8k_dataset_player_O")

# 4. Continue Model Train
# continue_model_train("minimax_O_logs_12k_dataset_player_O_model","minimax_O_logs_8k_dataset_player_O")

# 5. Predict
# board = [["X", "X", "O"],
#          ["X", "O", "O"],
#          ["-", "O", "-"]]
# model = load_model("logs_50k_dataset_player_X_model")
# move = predict_move_from_board(model, board)
# print("move :", move)
