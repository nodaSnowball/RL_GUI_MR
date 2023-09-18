# RL_GUI_MR
A interface with rearrange task simulation environment. Rearrangement is to reach the target room configuration through agent's actions.
## Preparation
You can install the required packages with commands below (or in the requirement.txt file):\
  conda install -c conda-forge pybullet\
  pip install pillow matplotlib gym opencv-python
## To start the game
Download the files and run game.py. A window asking for user name should pop up. Enter the name and you will see manual instruction and menu windows. Read the instruction first as it contains much important information about this game. There are 4 buttons in the menu window: manual instruction - used to open the instruction window anytime, train - used to open training interface of the game, eval - used to open evaluation interface, and lastly, end episode - used to end a game manualy but wont provide save result option as it can report a bug. After each game ending natrually, a window showing your total rewards should show up and then you can choose if you want to save the result of this game or not.
## Training Mode
In training mode, information and rewards will be shown to players. You can train yourself for the game according to the relationships between actions and rewards.
## Evaluating Mode
In evaluating mode, no information but steps you have taken will be shown. You need to utilize what you have learned during training phase to do the task.
## Instructions for the Game
Choose the game window (which has two images) and press the key to input actions. For detailed information, please refer to the manul instruction in the game.
## Saving Results
After the game, you can choose to save the result if you feel you have trained enough. If possible, please collect data throughout your training process and evaluate process. The result will be save to the same path as the 'game.py' script. A result folder wil be created and csv sheets containing the data will be inside.
