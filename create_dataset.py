"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### For model traning purpose ###

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import numpy as np

from ple import PLE

from ple.games.flappybird import FlappyBird as flappybird_game
from ple.games import base
from pygame.constants import K_w, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)	

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
nb_frames = 100000000000
step = 0
gamescores = 0.0
reward = 0.0
posibility_actions = [ 
[K_h, K_h, K_h, K_h, K_h, K_h, K_h, K_h],
[K_h, K_h, K_h, K_h, K_h, K_h, K_h, K_h],

[K_w, K_h, K_h, K_h, K_h, K_h, K_h, K_h],
[K_h, K_h, K_h, K_h, K_w, K_h, K_h, K_h],

[K_w, K_w, K_h, K_h, K_h, K_h, K_h, K_h],
[K_w, K_h, K_h, K_h, K_w, K_h, K_h, K_h],

[K_w, K_w, K_h, K_h, K_h, K_h, K_h, K_h],
[K_w, K_h, K_h, K_h, K_w, K_h, K_h, K_h],

[K_w, K_w, K_w, K_h, K_h, K_h, K_h, K_h],
[K_w, K_h, K_w, K_h, K_w, K_h, K_h, K_h]
]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def random_action( posibility_actions ): 
	rewards = 0.0
	
	gameState = p.getGameState()
	player_y_array = gameState['player_y']
	player_vel_array = gameState['player_vel']
	next_pipe_dist_to_player_array = gameState['next_pipe_dist_to_player']
	next_pipe_top_y_array = gameState['next_pipe_top_y']
	next_pipe_bottom_y_array = gameState['next_pipe_bottom_y']
	next_next_pipe_dist_to_player_array = gameState['next_next_pipe_dist_to_player']
	next_next_pipe_top_y_array = gameState['next_next_pipe_top_y']
	next_next_pipe_bottom_y_array = gameState['next_next_pipe_bottom_y']
	
	gap = (( next_pipe_bottom_y_array - next_pipe_top_y_array ) / 2 )
	top = next_pipe_top_y_array
	target = top + gap
	
	height_diff = player_y_array - next_pipe_top_y_array
	angle_diff = height_diff / ( next_pipe_dist_to_player_array + 1 )
	
	height_diff_2 = next_pipe_top_y_array - next_next_pipe_top_y_array
	angle_diff_2 = height_diff_2 / ( next_next_pipe_dist_to_player_array - next_pipe_dist_to_player_array + 1 )
	
	height_diff_3 = player_y_array - next_next_pipe_top_y_array
	angle_diff_3 = height_diff_3 / ( next_pipe_dist_to_player_array + 1 )
	
	height_diff_4 = next_pipe_top_y_array - next_next_pipe_bottom_y_array
	angle_diff_4 = height_diff_4 / ( next_next_pipe_dist_to_player_array - next_pipe_dist_to_player_array + 1 )
	
	angle_diff_5 = angle_diff_2 * angle_diff_3 / ( angle_diff + 1 )
	
	
	coefficient_0 = next_pipe_bottom_y_array - player_y_array + 0
	coefficient_1 = next_pipe_bottom_y_array - player_y_array + 5
	coefficient_2 = next_pipe_bottom_y_array - player_y_array + 10
	coefficient_3 = next_pipe_bottom_y_array - player_y_array + 30
	coefficient_4 = next_pipe_bottom_y_array - player_y_array + 40
	coefficient_5 = player_y_array - next_pipe_top_y_array + 0
	coefficient_6 = player_y_array - next_pipe_top_y_array + 5
	coefficient_7 = player_y_array - next_pipe_top_y_array + 10
	coefficient_8 = player_y_array - next_pipe_top_y_array + 15
	coefficient_9 = player_y_array - next_pipe_top_y_array + 20
	
	coefficient_0 = angle_diff_4 * 0.0090 * ( coefficient_0 - (player_y_array - target) ) - 5
	coefficient_1 = angle_diff_4 * 0.0085 * ( coefficient_1 - (player_y_array - target) )
	coefficient_2 = angle_diff_4 * 0.0080 * ( coefficient_2 - (player_y_array - target) ) - 5
	coefficient_3 = angle_diff_4 * 0.0075 * ( coefficient_3 - (player_y_array - target) )
	coefficient_4 = angle_diff_4 * 0.0070 * ( coefficient_4 - (player_y_array - target) ) - 5
	coefficient_5 = angle_diff_4 * 0.0065 * ( coefficient_5 - (player_y_array - target) )
	coefficient_6 = angle_diff_4 * 0.0060 * ( coefficient_6 - (player_y_array - target) ) - 5 
	coefficient_7 = angle_diff_4 * 0.0055 * ( coefficient_7 - (player_y_array - target) )
	coefficient_8 = angle_diff_4 * 0.0050 * ( coefficient_8 - (player_y_array - target) ) - 5
	coefficient_9 = angle_diff_4 * 0.0045 * ( coefficient_9 - (player_y_array - target) )
	
	temp = tf.random.normal([10], 1, 0.2, tf.float32)
	temp = np.asarray(temp) * np.asarray([ 	coefficient_0, coefficient_1, coefficient_2, coefficient_3, coefficient_4, coefficient_5, 
											coefficient_6, coefficient_7, coefficient_8, coefficient_9 ]) #action = actions['up']
	temp = tf.nn.softmax(temp)
	
	if ( angle_diff_4 > 0 ) :
		action = int(np.argmax(temp))	
	else :
		action = int(np.argmin(temp))
		
	for i in range(len(posibility_actions[0])) :
		reward = p.act(posibility_actions[action][i])
		
		if reward < 0 :
			return reward
		else :
			rewards = rewards + reward

	return rewards

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = flappybird_game(width=288, height=512, pipe_gap=250)
p = PLE(game_console, fps=30, display_screen=True)
p.init()

obs = p.getScreenRGB()	

for i in range(nb_frames):
	
	step = step + 1
	gamescores = gamescores + reward
	reward = 0
	
	if p.game_over():	
		step = 1
		gamescores = 0
		reward = 0	

		game_console = flappybird_game(width=288, height=512, pipe_gap=250)
		p = PLE(game_console, fps=30, display_screen=True)
		p.init()
		p.reset_game()
	
	if ( step == 1 ):
		print('start .... ' )
		
		for j in range(8):
			reward = p.act(K_h)
			reward = p.act(K_w)
		
		for j in range(15):
			reward = p.act(K_h)
			reward = p.act(K_h)
			
		for j in range(3):
			reward = p.act(K_h)
			reward = p.act(K_w)
	
	else :
		action = random_action( posibility_actions )
		input('...')
		
input('END !!!')
