# Remote_devices
Running parallel parameters on the remote devices

### Games input environment parameters ###

By the game support input parameters are game player locations and objects in the stage we using these parametes as input and convert into target trained parallel parameters to run in the remote devices.
```
gameState = p.getGameState()
player_y_array = gameState['player_y']
player_vel_array = gameState['player_vel']
next_pipe_dist_to_player_array = gameState['next_pipe_dist_to_player']
next_pipe_top_y_array = gameState['next_pipe_top_y']
next_pipe_bottom_y_array = gameState['next_pipe_bottom_y']
next_next_pipe_dist_to_player_array = gameState['next_next_pipe_dist_to_player']
next_next_pipe_top_y_array = gameState['next_next_pipe_top_y']
next_next_pipe_bottom_y_array = gameState['next_next_pipe_bottom_y']
```

### Actions and action as dictionaty ###
```
actions = { "up_1": K_h, "none_1": K_w, "none_2": K_h, "none_3": K_h, "none_4": K_h, "none_5": K_h, 
            "none_6": K_h, "none_7": K_h, "none_8": K_h, "none_9": K_h }
action = predict_action( )
reward = p.act(list(actions.values())[action])
```

### Running parallel parameters in remote devices ###

Using Tensorflow or any random function to running model training parallel parameters on remote devices, it is simly quantization and distribution in statistics by load parameters are the environment response, select one of the hight responsive scores as the game auto-play response.
```
temp = tf.random.normal([10], 1, 0.2, tf.float32)
temp = np.asarray(temp) * np.asarray([ coefficient_0, coefficient_1, coefficient_2, coefficient_3, 
coefficient_4, coefficient_5, coefficient_6, coefficient_7, coefficient_8, coefficient_9 ])
temp = tf.nn.softmax(temp)
action = int(np.argmax(temp))	
```


## Result image ##
![Alt text](https://github.com/jkaewprateep/Remote_devices/blob/main/FlappyBird_small.gif?raw=true "Title")
