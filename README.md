# Remote_devices
Running parallel parameters on the remote devices, usages of AI and machine learning are widely spreads and we cannot installed Tensorflow into every devices. The method we usually do it transfrom the final results from model trained layers into parallel parameters it is when you looking into the model weights it is only numbers output as arrays.

### Trained model weights or layer weights ###

Model weight is a set of trained parameters and bias those depends on what is your input into the model and layer, you can build the custom layers without bias but it will take long time to converges or it not converges for some optimizers depend on backward-propagation learning only. 
```
[[[[ 0.2707774  -0.06248572 -0.21273609]
   [-0.14883357 -0.25654262 -0.3296967 ]]

  [[ 0.08184502 -0.26611787 -0.0798333 ]
   [-0.03979453  0.10016242 -0.09921739]]

  [[ 0.14356184  0.3443761  -0.21177462]
   [-0.26943463  0.04512939 -0.18201178]]]
```

### Not leanrning layer, freezes answer layer ###

In some implementations they required the same results for everytime running such as calculation of units and values, summation and minmum-maximum or runing specific parameters input without training updates.
```
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs
		
	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
		shape=[int(input_shape[-1]),
		self.num_outputs])

	def call(self, inputs):
		return tf.matmul(inputs, self.kernel)
```

### Leanrning layer, tracing back update weights layer ###

In backward-propagation learning is the estimation of weights change of the previous input or result compares to current ( Weights + bias ) and the optimizer is engine to remarks targets and loss function is the estimation of the paces from current step to next as traveling path. The update back of the values compare to current weights and bias and update. Some algotithms update once or update everytime running, it is the same as people learning when the knowledge or senses is changed from they acceptable value ( only a bit amount 0.00001 as learning rates but it is not the learning rates it is different thing small value as that ) or the bias. The learning bias or they beleviving it is true and we have the new true or update information that dilivery only a bit of values updates to weights. The result in weights changed see as trained parameters and we finallzed as parallel parameters.
```
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs, name="MyDenseLayer"):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs

	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
		shape=[int(input_shape[-1]), 1])
		self.biases = tf.zeros([int(input_shape[-1]), 1])

	def call(self, inputs):
	
		# Weights from learning effects with input.
		temp = tf.reshape( inputs, shape=(10, 1) )
		temp = tf.matmul( inputs, self.kernel ) + self.biases
		
		# Posibility of x in all and x.
		return tf.nn.softmax( temp, axis=0 )
```

### Leanrning layer, extracting weights from trained layer ###

It is simply as call varaibles from a return functions, the output is numbers of arrays you can copy and saved or resue of it, simple mathametics required at this point because you need to match the output to remote target learning devices or diemensions extraction layers or create inverse matrix for parallel parameters as you see the final results we running the auto-play auto-pilots game Flappy birds and many of games by only simple functions that run in any platform or multiplexers.
```
Conv2DTranspose_01 = model.get_layer( name="Conv2DTranspose_01" )
weights_01 = Conv2DTranspose_01.get_weights()[0]
weights_02 = Conv2DTranspose_01.get_weights()[1]
```

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

I explained as distributions statistics when there are 10 actions target, 10 input parameters from game running and our parallel parameters from trained model and simple function of program such as MAX, MIN, MOD, DIV, SUB, ADD, MUL, RAND and etc. We use SoftMax() to help about understanding oterwise it required some conditions when the output values are too different and we compare them.
```
actions = { "up_1": K_h, "none_1": K_w, "none_2": K_h, "none_3": K_h, "none_4": K_h, "none_5": K_h, 
            "none_6": K_h, "none_7": K_h, "none_8": K_h, "none_9": K_h }
action = predict_action( )
reward = p.act(list(actions.values())[action])
```

### Running parallel parameters in remote devices ###

Using Tensorflow or any random function to running model training parallel parameters on remote devices, it is simly quantization and distribution in statistics by load parameters are the environment response, select one of the hight responsive scores as the game auto-play response. There are number of steps average from one gap to anoter gap as traveling path and player height and gap heights different. We do just select maximum number response for ranges setup and select action from the list action number return from function. 

### Line dot function for guides our Flappy bird play to fly ###

We create a guidline for out AI to learn about the game rules, that is important because AI not only self-learning but also learn follow by the rules we setup and we as referees we know it is learning as the objective or they try something else.
```
gap 0 = abs( player_y_array - next_pipe_top_y_array ) - saety value 1
gap 1 = abs( player_y_array - abs( next_pipe_top_y_array - next_pipe_bottom_y_array ) ) - saety value 2
gap 2 = abs( player_y_array - abs( next_pipe_top_y_array - next_pipe_bottom_y_array ) ) - saety value 3
gap 3 = abs( player_y_array - abs( next_pipe_top_y_array - next_pipe_bottom_y_array ) ) - saety value 4
gap 4 = abs( player_y_array - abs( next_pipe_top_y_array - next_pipe_bottom_y_array ) ) - saety value 5
gap 5 = abs( player_y_array - abs( next_next_pipe_top_y_array - next_next_pipe_bottom_y_array ) ) - saety value 6
gap 6 = abs( player_y_array - abs( next_next_pipe_top_y_array - next_next_pipe_bottom_y_array ) ) - saety value 7
gap 7 = abs( player_y_array - abs( next_next_pipe_top_y_array - next_next_pipe_bottom_y_array ) ) - saety value 8
gap 8 = abs( player_y_array - abs( next_next_pipe_top_y_array - next_next_pipe_bottom_y_array ) ) - saety value 9
gap 9 = abs( player_y_array - next_next_pipe_bottom_y_array ) - saety value 10
```

### Final running parallel parameters ###

At the remote deivices we train the model using model.fit() and we see the results is acceptable we extract weights values or we train a simetric model and extracting values to fit the target matrix capable. Those coeeficients are input from the game environment change on each step in the game state.
```
temp = tf.random.normal([10], 1, 0.2, tf.float32)
temp = np.asarray(temp) * np.asarray([ coefficient_0, coefficient_1, coefficient_2, coefficient_3, 
coefficient_4, coefficient_5, coefficient_6, coefficient_7, coefficient_8, coefficient_9 ])
temp = tf.nn.softmax(temp)
action = int(np.argmax(temp))	
```

### Files and Directory ###

Using Tensorflow to create dataset file, using for model training for parallel parameters.
``` 
1. create_dataset.py
2. FlappyBird_small.gif
3. README.md
``` 

## Result image ##
![Alt text](https://github.com/jkaewprateep/Remote_devices/blob/main/FlappyBird_small.gif?raw=true "Title")
