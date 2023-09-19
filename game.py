import os
import csv
import time
import tkinter as tk
import tkinter.messagebox
import tkinter.simpledialog
import tkinter.ttk as ttk
from tkinter import *
from PIL import Image, ImageTk
import json
import time

from box_world_decision_GT import BoxWorld


class HumanPlay:
    def __init__(self):

        self.user = 'admin'

        self.session_id = str(time.time())[-6:]
        self.intro_info = "This is a human interface for rearrangement task in which you can control the robot and\n" \
                          "interact with the environment. The keys and their bonded actions pairs you can use are:\n" \
                          " <up>:   forward\n<left>:   turn left\n<right>:   turn right\n<d>:      pick\n<f>:     drop\nThe " \
                          "interface will show two images: on the left it is a first person of view (fpv) image of" \
                          " the robot;\nanother is the target room configuration that we want to achieve. Your task " \
                          "is to control the robot\nwith the fpv image and move the object in the environment to " \
                          "reach the target room configuration.\nAt the beginning of each game, the agent will always" \
                          " be reset to the center and toward to the\nleft (initial orientation)\n\nUtilize the " \
                          "actions above and you can open" \
                          " this note any time through menu window. \n\nAbout saving results: after each game, you " \
                          "can choose to save the result, but ending a game\nmanually will skip this step. " \
                          "Thus, while collecting results, please " \
                          "keep input actions until the\ngame ends by itself. The maximum step of a game is 1000. " \
                          "Thanks for your patience and hope\nyou enjoy this!" \
                          ""

        # Env
        self.ENV_DICT = {
            1: 'Polygon',
            2: 'Rectangle',
        }
        self.act2idx = {
            'F': 0,
            'L': 1,
            'R': 2,
            'pick': 3,
            'drop': 4
        }
        self.env = None
        self.env_choose = 2  # goes from 1-12 (1D, 2D and 3D plans supported)
        self.environment_width = None
        self.environment_height = None
        self.window_size = None
        self.max_steps = None
        self.max_bricks = None

        self.action_idx = None
        self.L_action_idx = None
        self.R_action_idx = None
        self.U_action_idx = None
        self.D_action_idx = None
        self.L_brick_action_idx = None
        self.R_brick_action_idx = None
        self.U_brick_action_idx = None
        self.D_brick_action_idx = None

        # State
        self.step_count = 0
        self.brick_count = 0
        self.reward = 0
        self.episode_reward = 0
        self.environment_memory = None
        self.observation = None
        self.fpv = None
        self.target_bev = None
        self.fpv_label = None
        self.target_bev_label = None
        self.done = None
        self.info = None

        # GUI
        self.info_window = None
        self.menu_window = Tk()
        self.user = tk.simpledialog.askstring('Enter user', 'Enter your first name (all-lowercase):')
        self.custom_message_box()
        self.menu_window.title('Menu')
        x = (self.menu_window.winfo_screenwidth() - self.menu_window.winfo_reqwidth()) / 2
        y = (self.menu_window.winfo_screenheight() - self.menu_window.winfo_reqheight()) / 2
        self.menu_window.geometry('+%d+%d' % (x, y))
        self.game_window = None
        self.game_mode = None
        self.play_again_window = None

        self.plan_window = None
        self.plan_canvas = None

        self.canvas = None
        self.canvas_width = None
        self.canvas_height = None
        self.num_steps_id = None
        self.num_reward_id = None
        self.num_totalreward_id = None

        self.build_menu_window()
        # self.reset_game(self.env_choose)

        # fig = plt.figure(figsize=[5, 5])
        # fig = plt.figure()
        # self.axe = fig.add_subplot(1, 1, 1, projection='3d')

    def custom_message_box(self):
        if self.info_window is not None:
            self.info_window.destroy()
        self.info_window = tk.Toplevel()
        self.info_window.title("Manual Instruction")
        x = (self.info_window.winfo_screenwidth() - self.info_window.winfo_reqwidth()) / 2
        y = (self.info_window.winfo_screenheight() - self.info_window.winfo_reqheight()) / 2
        self.info_window.geometry('+%d+%d' % (x, y))

        # Define a custom font
        custom_font = ("Arial", 14)  # Replace with your desired font and size

        # Create a Label widget with the custom font for the message
        message_label = tk.Label(self.info_window, text=self.intro_info, font=custom_font)
        message_label.pack(padx=20, pady=10)

        # Create a "OK" button to close the custom message box
        ok_button = tk.Button(self.info_window, text="OK", command=self.info_window.destroy)
        ok_button.pack(pady=10)

    def quit(self):
        if self.game_window is not None:
            self.game_window.destroy()
        if self.play_again_window is not None:
            self.play_again_window.destroy()

    def exit(self):
        if self.game_window is not None:
            self.game_window.destroy()
        if self.play_again_window is not None:
            self.play_again_window.destroy()
        self.menu_window.destroy()

    def play_again_box(self):
        self.play_again_window = tk.Toplevel()
        self.play_again_window.title("Complete")
        x = (self.play_again_window.winfo_screenwidth() - self.play_again_window.winfo_reqwidth()) / 2
        y = (self.play_again_window.winfo_screenheight() - self.play_again_window.winfo_reqheight()) / 2
        self.play_again_window.geometry('+%d+%d' % (x, y))

        # Define a custom font
        custom_font = ("Arial", 14)  # Replace with your desired font and size

        # Create a Label widget with the custom font for the message
        message_label = tk.Label(self.play_again_window, text='Do you want to play again?', font=custom_font)
        message_label.pack(padx=20, pady=10)

        # Create an "OK" button to close the custom message box
        tk.Button(self.play_again_window, width=20, text="Play again", command=self.reset_game).pack()
        tk.Button(self.play_again_window, width=20, text="Quit", command=self.quit).pack()

    def reset_game(self, mode=None):
        if self.play_again_window:
            self.play_again_window.destroy()
        self.env = BoxWorld(max_steps_per_episode=1000, isGUI=False)
        self.env.reset()

        if mode == 1:
            self.game_mode = 'train'
        elif mode == 2:
            self.game_mode = 'eval'
        self.step_count = 0
        self.brick_count = 0
        self.reward = 0
        self.info = None
        self.episode_reward = 0
        self.done = False
        self.done = False
        self.data = []

        self.display_state = [0 for i in range(5)]
        self.canvas_width = 1300
        self.canvas_height = 700

        self.idx_left = 0
        self.idx_right = 1
        self.idx_move_forward = 2
        self.idx_pick = 3
        self.idx_drop = 4

        self.observation = self.env.gui_output()
        self.target_bev = self.env.get_target_bev()
        self.reset_game_window(mode)
        self.update_canvas()

    def build_menu_window(self):
        font = ('Arial', 14)
        self.game_mode = 'train'  # 1 for train, 2 for eval

        tk.Button(self.menu_window, width=20, font=font, text="Manual Instruction", fg="red",
                  command=self.custom_message_box).pack()
        tk.Button(self.menu_window, width=20, font=font, text="Train", fg="blue",
                  command=lambda x=1: self.reset_game(x)).pack()
        tk.Button(self.menu_window, width=20, font=font, text="Evaluate", fg="blue",
                  command=lambda x=2: self.reset_game(x)).pack()
        tk.Button(self.menu_window, width=20, font=font, text="Exit", fg="black", command=self.exit).pack()

    def reset_game_window(self, mode):
        if self.game_window is not None:
            self.game_window.destroy()

        self.game_window = Toplevel()
        self.game_window.geometry('+500+500')
        self.game_window.title('Human Performance Benchmarking')
        tk.Button(self.game_window, text='Quit', font=('Arial', 14), fg='black',
                  command=self.upon_episode_completion).pack(side='bottom')
        self.game_window.bind("<Key>", self.keypress)
        self.game_window.bind("<Left>", lambda event: self.move('L'))
        self.game_window.bind("<Right>", lambda event: self.move('R'))
        self.game_window.bind("<Up>", lambda event: self.move('F'))
        self.game_window.bind("<d>", lambda event: self.move('pick'))
        self.game_window.bind("<f>", lambda event: self.move('drop'))

        self.canvas = Canvas(
            self.game_window,
            width=self.canvas_width,
            height=self.canvas_height)
        self.canvas.pack()

        self.canvas.create_rectangle(150 - 1, 50 - 1, 550 + 1, 450 + 1, outline='gray75')
        self.canvas.create_rectangle(750 - 1, 50 - 1, 1150 + 1, 450 + 1, outline='gray75')

        self.txt_plan_id = self.canvas.create_text(650, 680, text='Mobile Rearrangement Interface',
                                                   fill='gray60', font=('Arial', 10))
        self.txt_img1 = self.canvas.create_text(300, 500, text='FPV of Agent',
                                                fill='gray60', font=('Arial', 10))
        self.txt_img2 = self.canvas.create_text(950, 500, text='Target BEV',
                                                fill='gray60', font=('Arial', 10))
        self.fpv = self.observation
        self.fpv = ImageTk.PhotoImage(Image.fromarray(self.fpv).resize((400, 400)))
        self.target_bev = ImageTk.PhotoImage(Image.fromarray(self.target_bev).resize((400, 400)))

        self.fpv_label = self.canvas.create_image(150, 50, anchor=tk.NW, image=self.fpv)
        self.target_bev_label = self.canvas.create_image(750, 50, anchor=tk.NW, image=self.target_bev)

        self.line1 = self.canvas.create_line(850, 250, 950, 250, fill="red", width=2)
        self.line2 = self.canvas.create_line(850, 250, 870, 260, fill="red", width=2)
        self.line3 = self.canvas.create_line(850, 250, 870, 240, fill="red", width=2)
        self.description = self.canvas.create_text(900, 230, text='initial orientation', fill='red')
        # self.exit = self.canvas.create_text(650, 30, text='press ESC to end this game', fill='red',
        #                                     font=('Arial', 15))


        if self.in_training_mode():
            self.txt_steps_id = self.canvas.create_text(100, 570, text='steps taken',
                                                        fill='gray60', font=('Arial', 12))
            self.num_steps_id = self.canvas.create_text(100, 600, text=str(0),
                                                        fill='gray40', font=('Arial', 18))
            self.txt_reward_id = self.canvas.create_text(300, 570, text='step reward',
                                                         fill='gray60', font=('Arial', 12))
            self.txt_totalreward_id = self.canvas.create_text(500, 570, text='total reward',
                                                              fill='gray60', font=('Arial', 12))
            self.num_reward_id = self.canvas.create_text(300, 600, text=str(0),
                                                         fill='gray40', font=('Arial', 18))
            self.num_totalreward_id = self.canvas.create_text(500, 600, text=str(0),
                                                              fill='gray40', font=('Arial', 18))
            self.txt_info_id = self.canvas.create_text(1000, 570, text='info',
                                                       fill='gray60', font=('Arial', 12))
            self.content_info_id = self.canvas.create_text(1000, 600, text='The robot has been reset to the center of'
                                                                           'the map and toward to the left,\nplease '
                                                                           'take your first step',
                                                           fill='gray40', font=('Arial', 12))
            self.txt_totalreward_id = self.canvas.create_text(500, 570, text=self.info,
                                                              fill='gray60', font=('Arial', 12))
        else:
            self.txt_steps_id = self.canvas.create_text(650, 570, text='steps taken',
                                                        fill='gray60', font=('Arial', 12))
            self.num_steps_id = self.canvas.create_text(650, 600, text=str(0),
                                                        fill='gray40', font=('Arial', 18))

    def update_canvas(self):
        self.fpv = self.observation
        self.fpv = ImageTk.PhotoImage(Image.fromarray(self.fpv).resize((400, 400)))
        self.canvas.itemconfig(self.fpv_label, image=self.fpv)

        self.canvas.itemconfig(self.num_steps_id, text=str(self.step_count))

        if self.in_training_mode():
            self.canvas.itemconfig(self.content_info_id, text=self.info)
            self.canvas.itemconfig(self.num_reward_id, text=str(self.reward))
            self.canvas.itemconfig(self.num_totalreward_id, text=str(self.episode_reward))

    def upon_episode_completion(self):
        save = tkinter.messagebox.askquestion("Save", f"Episode ended, total reward of the game: {self.episode_reward}"
                                                      f"\n\t- save the result?")

        if save == 'yes':
            if self.game_mode == 'train':
                path = 'results/human_train_results' + '_' + self.user + '_' + self.session_id + '.csv'
            else:
                path = 'results/human_eval_results' + '_' + self.user + '_' + self.session_id + '.csv'
            self.log_result(path)
            tkinter.messagebox.showinfo("Saved.", "Episode results appended to:\n" + os.path.abspath(path))
        else:
            tkinter.messagebox.showinfo("Not saved.", "Episode results discarded.")

        self.play_again_box()

    def log_result(self, path):

        if not os.path.exists('results'):
            os.makedirs('results')

        tick = time.time()
        savename = str(self.user) + '_' + str(tick)
        result = self.env.get_result()

        with open(path, 'a', newline='') as log_file:
            schema = ['user', 'game_mode', 'num_steps', 'success', 'fixed_strict', 'Energy_Remaining']
            writer = csv.DictWriter(log_file, fieldnames=schema)
            writer.writerow({'user': self.user,
                             'game_mode': self.game_mode,
                             'num_steps': self.step_count,
                             'success': result[0],
                             'fixed_strict': result[1],
                             'Energy_Remaining': result[2]})

    def move(self, action):
        action = self.act2idx[action]
        self.data.append(action)
        _, self.reward, self.done, self.info = self.env.step(action)
        self.observation = self.env.gui_output()
        self.step_count += 1
        self.episode_reward += self.reward
        self.after_move()

    def after_move(self):
        self.update_canvas()
        if self.done:
            time.sleep(0.5)
            self.upon_episode_completion()

    def keypress(self, event):
        if event.char == ' ' and self.env_choose <= 8:  # Space bar (drop brick in current location)
            self.observation, self.reward, self.done = self.env.step(self.action_idx)
            self.episode_reward += self.reward
            self.step_count += 1
            self.after_move()


        elif event.char == '\x1b':  # Escape key (stop playing)
            self.upon_episode_completion()
            self.game_window.destroy()
            self.game_window = None

    def in_training_mode(self):
        return self.game_mode == "train"

    def set_mode(self, event):
        if self.in_training_mode():
            self.reset_game(self.env_choose)
        else:
            self.reset_game(self.env_choose)

    def mainloop(self):
        self.menu_window.mainloop()


game = HumanPlay()
game.mainloop()
