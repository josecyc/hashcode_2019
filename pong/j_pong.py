import numpy as np
import _pickle as pickle
import gym

#hyperparameters
H = 200
batch_size = 10 #every how many episodes and update is done
learning_rate = .0004
gamma = 0.99 #reward discount factor
decay_rate = 0.99 #decay factor for RMSProp 
#resume = False
#render = False

#model initialization
D = 80 * 80 #row size of input column of pixels
model = {}
model['W1'] = np.random.randn(H, D) / np.sqrt(D) #Xavier initialization, "just right size"
model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k, v in model.items() }
#grad_buffer = What is this?
rmsprop_cache = { k : np.zeros_like(v) for k, v in model.items() }
#rmsprop_cache = What is this?

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x)) #sigmoid squashing function to interval [0,1]

def prepro(I):
	#original 210 rows 160 columns 3 colors 
	#preprocessing from a lot of pixels to 6400 1D vector
	I = I[35:195] #cut from row 35 to row 195
	I = I[::2, ::2, 0] #get only every 2nd pixel? i.e. downsample by a factor of 2
	I[I == 144] = 0 #erases background type; making it a certain color?
	I[I == 109] = 0 #erases another background type 
	I[I != 0] = 1 #set all other pixels (paddles, ball) to 1
	return I.astype(np.float).ravel()

def discount_rewards(r):
	"""take 1D float array of REWARDS [0 0 0 ... 1 ...] and compute 
	discounted reward
	This calculates the discounted reward for every reward, meaning
	every future reward has a lower value relative to the current
	one.
	"""
	discounted_r = np.zeros_like(r)
	running_addition = 0
	for t in reversed(range(0, r.size)):
		if r[t] != 0: running_addition = 0 #reset sum 
		#reset the sum, since this was a game boundary (pong specific!)
		running_addition = running_addition * gamma + r[t] 
		#R * .99 + r  
		discounter_r[t] = running_addition
	return discounted_r

def policy_forward(x):
	h = np.dot(model['W1'], x)
	h[h<0] = 0 #Relu nonlinearity
	logp = np.dot(model['W1'], h)
        p = sigmoid(logp)
        return p, h

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0) # only difference
	
def build_graph(observations):
    """Calculates logits from the input observations tensor.
        This function will be called twice: rollout and train.
	    The weights will be shared.
	        """
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE):	        
            hidden = tf.layers.dense(observations, args.hidden_dim, use_bias=False, activation=tf.nn.relu)
	    logits = tf.layers.dense(hidden, len(ACTIONS), use_bias=False)
            return logits

def policy_backward(eph, epdlogp):
    """Backward pass, eph is array of intermediate hidden states"""
    #eph == array of hidden states, i.e. array of hs?
    #edplogp == 
    dW2 = np.dot(eph.T, edplogp).ravel()
    #Why is the derivative of W2 the dot product of eph.t and edplogp??
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 #backprop with ReLu
    dW1 = np.dot(dh.T, epx)
    #dh.T = 
    #epx =
    return {'W1':dW1, 'W2':dW2}




