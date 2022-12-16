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

### Files and Directory ###

Using Tensorflow to create dataset file, using for model training for parallel parameters.
``` 
1. create_dataset.py
2. FlappyBird_small.gif
3. README.md
``` 

## Result image ##
![Alt text](https://github.com/jkaewprateep/Remote_devices/blob/main/FlappyBird_small.gif?raw=true "Title")
