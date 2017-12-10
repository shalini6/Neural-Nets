import numpy as np 

class NeuralNetwork():
	def __init__(self):
		np.random.seed(1)
		self.weights = 2*np.random.random((3,1))-1

	def __sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def __sigmoid_derivative(self,x):
		return x*(1-x)

	def predict(self,inputs):
		return self.__sigmoid(np.dot(inputs,self.weights))

	def train(self,train_in,train_out,iters):
		for i in xrange(iters):
			output = self.predict(train_in)
			error = train_out - output
			delta = np.dot(train_in.T,error*self.__sigmoid_derivative(output))
			self.weights += delta


if __name__ == '__main__':
	neural_net = NeuralNetwork()
	print 'Random starting weights:'
	print neural_net.weights
	train_in = np.array([[0,0,1],[1,1,1],[1,0,1],[1,1,0]])
	train_out = np.array([[1,1,0,0]]).T 
	neural_net.train(train_in,train_out,10000)

	print 'New weights after training:'
	print neural_net.weights

	print 'Running Predictions'
	print neural_net.predict(np.array([0,1,0]))