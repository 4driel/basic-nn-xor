import numpy as np

class neuralNetwork(object):
	def __init__(self):
		#Hyper params
		self.inputUnits = 2
		self.hiddenUnits = 3
		self.outputUnits = 1

		self.learningRate = 1
		self.iterations = 7000

		#Weights
		self.W1 = np.random.randn(self.inputUnits, self.hiddenUnits)
		self.W2 = np.random.randn(self.hiddenUnits, self.outputUnits)

	def forward(self, x):
		x = np.reshape(x, (-1,2))

		self.hiddenSum = np.dot(x, self.W1)
		self.hiddenResult = self.sigmoid(self.hiddenSum)
		self.outputSum = np.dot(self.hiddenResult, self.W2)
		self.outputResult = self.sigmoid(self.outputSum)
		return self.outputResult

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		return np.exp(-z)/((1 + np.exp(-z))**2)

	def errorFunction(self, x, y):
		self.forward(x)
        	return 0.5*sum((y-self.outputResult)**2)

	def backward(self, x, y, learn=True):
		x = np.reshape(x, (-1,2))
		y = np.reshape(y, (-1,1))

		self.delta3 = -(y-self.outputResult)*self.sigmoidPrime(self.outputSum)
		self.W2Changes = np.dot(self.hiddenResult.T, self.delta3)

		self.delta2 = np.dot(self.delta3, self.W2.T)*self.sigmoidPrime(self.hiddenSum)
		self.W1Changes = np.dot(x.T, self.delta2) 

		if learn == True:
			self.W2 = self.W2 - (self.learningRate * self.W2Changes)
			self.W1 = self.W1 - (self.learningRate * self.W1Changes)

	def computeGradients(self, x, y):
		self.forward(x)
		self.backward(x, y, learn=False)
		return np.concatenate((self.W1Changes.ravel(), self.W2Changes.ravel()))

	def computeNumGradients(self, x, y):
		#Flatten Weights into vector
		flattenedW = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		W1_end = self.hiddenUnits * self.inputUnits
		W2_end = W1_end + self.hiddenUnits * self.outputUnits

		#Create num gradient vector and perturb vector
		numGrad = np.zeros(flattenedW.shape)
		perturb = np.zeros(flattenedW.shape)
		e = 1e-4

		for p in range(len(flattenedW)):
			perturb[p] = e

			pertGrad = flattenedW + perturb
			self.W1 = np.reshape(pertGrad[0:W1_end], (self.inputUnits, self.hiddenUnits))
			self.W2 = np.reshape(pertGrad[W1_end:W2_end], (self.hiddenUnits, self.outputUnits))
			loss2 = self.errorFunction(x, y)

			pertGrad = flattenedW - perturb
			self.W1 = np.reshape(pertGrad[0:W1_end], (self.inputUnits, self.hiddenUnits))
			self.W2 = np.reshape(pertGrad[W1_end:W2_end], (self.hiddenUnits, self.outputUnits))
			loss1 = self.errorFunction(x, y)

			numGrad[p] = (loss2 - loss1)/ (2*e)
			perturb[p] = 0

		self.W1 = np.reshape(flattenedW[0:W1_end], (self.inputUnits, self.hiddenUnits))
		self.W2 = np.reshape(flattenedW[W1_end:W2_end], (self.hiddenUnits, self.outputUnits))
		return numGrad


#######################################################################################	

x = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=float)
y = np.array(([0],[1],[1],[0]), dtype=float)

nn = neuralNetwork()

print "training network..."
print "%d iterations" % (nn.iterations)

for i in range(0, nn.iterations):
	nn.forward(x)
	nn.backward(x, y)

print "y"
print y

print "y aprox"
print nn.outputResult

print "W2"
print nn.W2

print "W1"
print nn.W1
