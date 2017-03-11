import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
import sys
import cv2,csv

##Loading the data+
def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`"""
	if kind=='train':
		labels_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\train-labels.idx1-ubyte')		
		images_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\train-images.idx3-ubyte')
	else:
		labels_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\t10k-labels.idx1-ubyte')		
		images_path=os.path.abspath('D:\\pycode35\\AI\\mnist\\t10k-images.idx3-ubyte')
	
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II',
								 lbpath.read(8))
		labels = np.fromfile(lbpath,
							 dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII",
											   imgpath.read(16))
		images = np.fromfile(imgpath,
							 dtype=np.uint8).reshape(len(labels), 784)

	return images, labels

##Loading the data-
#儲存wieght
def saveweight(w1,w2):
	with open ('weight01.csv', mode='w',newline="\n") as write_file:
		writer = csv.writer(write_file)
		for i in range(len(w1)):
			writer.writerow([w1[i]])
	with open ('weight02.csv', mode='w',newline="\n") as write_file2:
		writer = csv.writer(write_file2)
		for i in range(len(w2)):
			writer.writerow([w2[i]])		
			
			
def loadWeight1():  
	l=[]  
	with open('weight01.csv') as file:	 
		 lines=csv.reader(file)	 
		 for line in lines:	 
			 l.append(line) 
	l=np.array(l).astype(float)
	data=l.copy()
	return data

def loadWeight2():  
	l=[]  
	with open('weight02.csv') as file:	 
		 lines=csv.reader(file)	 
		 for line in lines:	 
			 l.append(line) 
	l=np.array(l).astype(float)
	data=l.copy()
	return data 
	
##NeuralNetMLP+
class NeuralNetMLP(object):
	""" Feedforward neural network / Multi-layer perceptron classifier.

	Parameters
	------------
	n_output : int
		Number of output units, should be equal to the
		number of unique class labels.
	n_features : int
		Number of features (dimensions) in the target dataset.
		Should be equal to the number of columns in the X array.
	n_hidden : int (default: 30)
		Number of hidden units.
	l1 : float (default: 0.0)
		Lambda value for L1-regularization.
		No regularization if l1=0.0 (default)
	l2 : float (default: 0.0)
		Lambda value for L2-regularization.
		No regularization if l2=0.0 (default)
	epochs : int (default: 500)
		Number of passes over the training set.
	eta : float (default: 0.001)
		Learning rate.
	alpha : float (default: 0.0)
		Momentum constant. Factor multiplied with the
		gradient of the previous epoch t-1 to improve
		learning speed
		w(t) := w(t) - (grad(t) + alpha*grad(t-1))
	decrease_const : float (default: 0.0)
		Decrease constant. Shrinks the learning rate
		after each epoch via eta / (1 + epoch*decrease_const)
	shuffle : bool (default: True)
		Shuffles training data every epoch if True to prevent circles.
	minibatches : int (default: 1)
		Divides training data into k minibatches for efficiency.
		Normal gradient descent learning if k=1 (default).
	random_state : int (default: None)
		Set random state for shuffling and initializing the weights.

	Attributes
	-----------
	cost_ : list
	  Sum of squared errors after each epoch.

	"""
	def __init__(self, n_output, n_features, n_hidden=30,
				 l1=0.0, l2=0.0, epochs=500, eta=0.001,
				 alpha=0.0, decrease_const=0.0, shuffle=True,
				 minibatches=1, random_state=None):

		np.random.seed(random_state)
		self.n_output = n_output
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.w1, self.w2 = self._initialize_weights()
		self.l1 = l1
		self.l2 = l2
		self.epochs = epochs
		self.eta = eta
		self.alpha = alpha
		self.decrease_const = decrease_const
		self.shuffle = shuffle
		self.minibatches = minibatches

	def _encode_labels(self, y, k):
		"""Encode labels into one-hot representation

		Parameters
		------------
		y : array, shape = [n_samples]
			Target values.

		Returns
		-----------
		onehot : array, shape = (n_labels, n_samples)

		"""
		onehot = np.zeros((k, y.shape[0]))
		for idx, val in enumerate(y):
			onehot[val, idx] = 1.0
		return onehot

	def _initialize_weights(self):
		"""Initialize weights with small random numbers."""
		w1 = np.random.uniform(-1.0, 1.0,
							   size=self.n_hidden*(self.n_features + 1))
		w1 = w1.reshape(self.n_hidden, self.n_features + 1)
		w2 = np.random.uniform(-1.0, 1.0,
							   size=self.n_output*(self.n_hidden + 1))
		w2 = w2.reshape(self.n_output, self.n_hidden + 1)
		return w1, w2

	def _sigmoid(self, z):
		"""Compute logistic function (sigmoid)

		Uses scipy.special.expit to avoid overflow
		error for very small input values z.

		"""
		# return 1.0 / (1.0 + np.exp(-z))
		return expit(z)

	def _sigmoid_gradient(self, z):
		"""Compute gradient of the logistic function"""
		sg = self._sigmoid(z)
		return sg * (1.0 - sg)

	def _add_bias_unit(self, X, how='column'):
		"""Add bias unit (column or row of 1s) to array at index 0"""
		if how == 'column':
			X_new = np.ones((X.shape[0], X.shape[1] + 1))
			X_new[:, 1:] = X
		elif how == 'row':
			X_new = np.ones((X.shape[0] + 1, X.shape[1]))
			X_new[1:, :] = X
		else:
			raise AttributeError('`how` must be `column` or `row`')
		return X_new

	def _feedforward(self, X, w1, w2):
		"""Compute feedforward step

		Parameters
		-----------
		X : array, shape = [n_samples, n_features]
			Input layer with original features.
		w1 : array, shape = [n_hidden_units, n_features]
			Weight matrix for input layer -> hidden layer.
		w2 : array, shape = [n_output_units, n_hidden_units]
			Weight matrix for hidden layer -> output layer.

		Returns
		----------
		a1 : array, shape = [n_samples, n_features+1]
			Input values with bias unit.
		z2 : array, shape = [n_hidden, n_samples]
			Net input of hidden layer.
		a2 : array, shape = [n_hidden+1, n_samples]
			Activation of hidden layer.
		z3 : array, shape = [n_output_units, n_samples]
			Net input of output layer.
		a3 : array, shape = [n_output_units, n_samples]
			Activation of output layer.

		"""
		a1 = self._add_bias_unit(X, how='column')
		z2 = w1.dot(a1.T)
		a2 = self._sigmoid(z2)
		a2 = self._add_bias_unit(a2, how='row')
		z3 = w2.dot(a2)
		a3 = self._sigmoid(z3)
		return a1, z2, a2, z3, a3

	def _L2_reg(self, lambda_, w1, w2):
		"""Compute L2-regularization cost"""
		return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) +
								np.sum(w2[:, 1:] ** 2))

	def _L1_reg(self, lambda_, w1, w2):
		"""Compute L1-regularization cost"""
		return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() +
								np.abs(w2[:, 1:]).sum())

	def _get_cost(self, y_enc, output, w1, w2):
		"""Compute cost function.

		Parameters
		----------
		y_enc : array, shape = (n_labels, n_samples)
			one-hot encoded class labels.
		output : array, shape = [n_output_units, n_samples]
			Activation of the output layer (feedforward)
		w1 : array, shape = [n_hidden_units, n_features]
			Weight matrix for input layer -> hidden layer.
		w2 : array, shape = [n_output_units, n_hidden_units]
			Weight matrix for hidden layer -> output layer.

		Returns
		---------
		cost : float
			Regularized cost.

		"""
		term1 = -y_enc * (np.log(output))
		term2 = (1.0 - y_enc) * np.log(1.0 - output)
		cost = np.sum(term1 - term2)
		L1_term = self._L1_reg(self.l1, w1, w2)
		L2_term = self._L2_reg(self.l2, w1, w2)
		cost = cost + L1_term + L2_term
		return cost

	def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
		""" Compute gradient step using backpropagation.

		Parameters
		------------
		a1 : array, shape = [n_samples, n_features+1]
			Input values with bias unit.
		a2 : array, shape = [n_hidden+1, n_samples]
			Activation of hidden layer.
		a3 : array, shape = [n_output_units, n_samples]
			Activation of output layer.
		z2 : array, shape = [n_hidden, n_samples]
			Net input of hidden layer.
		y_enc : array, shape = (n_labels, n_samples)
			one-hot encoded class labels.
		w1 : array, shape = [n_hidden_units, n_features]
			Weight matrix for input layer -> hidden layer.
		w2 : array, shape = [n_output_units, n_hidden_units]
			Weight matrix for hidden layer -> output layer.

		Returns
		---------
		grad1 : array, shape = [n_hidden_units, n_features]
			Gradient of the weight matrix w1.
		grad2 : array, shape = [n_output_units, n_hidden_units]
			Gradient of the weight matrix w2.

		"""
		# backpropagation
		sigma3 = a3 - y_enc
		z2 = self._add_bias_unit(z2, how='row')
		sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
		sigma2 = sigma2[1:, :]
		grad1 = sigma2.dot(a1)
		grad2 = sigma3.dot(a2.T)

		# regularize
		grad1[:, 1:] += self.l2 * w1[:, 1:]
		grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
		grad2[:, 1:] += self.l2 * w2[:, 1:]
		grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

		return grad1, grad2

	def predict(self, X):
		"""Predict class labels

		Parameters
		-----------
		X : array, shape = [n_samples, n_features]
			Input layer with original features.

		Returns:
		----------
		y_pred : array, shape = [n_samples]
			Predicted class labels.

		"""
		if len(X.shape) != 2:
			raise AttributeError('X must be a [n_samples, n_features] array.\n'
								 'Use X[:,None] for 1-feature classification,'
								 '\nor X[[i]] for 1-sample classification')

		a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
		y_pred = np.argmax(z3, axis=0)
		return y_pred
		
	def predict2(self, X,W1,W2):
		"""Predict class labels

		Parameters
		-----------
		X : array, shape = [n_samples, n_features]
			Input layer with original features.

		Returns:
		----------
		y_pred : array, shape = [n_samples]
			Predicted class labels.

		"""
		if len(X.shape) != 2:
			raise AttributeError('X must be a [n_samples, n_features] array.\n'
								 'Use X[:,None] for 1-feature classification,'
								 '\nor X[[i]] for 1-sample classification')

		a1, z2, a2, z3, a3 = self._feedforward(X, W1, W2)
		y_pred = np.argmax(z3, axis=0) #argmax返回最大值索引值 ,0-9機率最高的那個
		return y_pred	

	def fit(self, X, y, print_progress=False):
		""" Learn weights from training data.

		Parameters
		-----------
		X : array, shape = [n_samples, n_features]
			Input layer with original features.
		y : array, shape = [n_samples]
			Target class labels.
		print_progress : bool (default: False)
			Prints progress as the number of epochs
			to stderr.

		Returns:
		----------
		self

		"""
		self.cost_ = []
		X_data, y_data = X.copy(), y.copy()
		y_enc = self._encode_labels(y, self.n_output)

		delta_w1_prev = np.zeros(self.w1.shape)
		delta_w2_prev = np.zeros(self.w2.shape)

		for i in range(self.epochs):

			# adaptive learning rate
			self.eta /= (1 + self.decrease_const*i)

			if print_progress:
				sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
				sys.stderr.flush()

			if self.shuffle:
				idx = np.random.permutation(y_data.shape[0])
				X_data, y_enc = X_data[idx], y_enc[:, idx]

			mini = np.array_split(range(y_data.shape[0]), self.minibatches)
			for idx in mini:

				# feedforward
				a1, z2, a2, z3, a3 = self._feedforward(X_data[idx],
													   self.w1,
													   self.w2)
				cost = self._get_cost(y_enc=y_enc[:, idx],
									  output=a3,
									  w1=self.w1,
									  w2=self.w2)
				self.cost_.append(cost)

				# compute gradient via backpropagation
				grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
												  a3=a3, z2=z2,
												  y_enc=y_enc[:, idx],
												  w1=self.w1,
												  w2=self.w2)

				delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
				self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
				self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
				delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

		return self
##NeuralNetMLP-

#load mnist data
X_train, y_train = load_mnist('mnist', kind='train')   #X_train=60000x784
print("train_data.shape=",X_train.shape)
X_test, y_test = load_mnist('mnist', kind='t10k')					 #X_test=10000x784
print("test_data.shape=",X_test.shape)

##宣告NeuralNetMLP物件 
nn = NeuralNetMLP(n_output=10, 
				  n_features=X_train.shape[1], 
				  n_hidden=100, 
				  l2=0.5, 
				  l1=0.0, 
				  epochs=1000, 
				  eta=0.002,
				  alpha=0.001,
				  decrease_const=0.00001,
				  minibatches=100, 
				  shuffle=True,
				  random_state=1)
				  
#False :跳過traning步驟直接使用已訓練儲存好的權重 
#True: 重新訓練權重並存檔				  
train_flag=True  
				  
if train_flag==True:
	nn.fit(X_train, y_train, print_progress=True)

	#y_train_pred = nn.predict(X_train)
	y_train_pred = nn.predict2(X_train,nn.w1,nn.w2)
	acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
	print('Training accuracy: %.2f%%' % (acc * 100))

	#儲存訓練完成的權重
	ww1=nn.w1.flatten()
	ww2=nn.w2.flatten()		
	saveweight(ww1,ww2)

	#畫出cost
	batches = np.array_split(range(len(nn.cost_)), 1000)
	cost_ary = np.array(nn.cost_)
	cost_avgs = [np.mean(cost_ary[i]) for i in batches]
	plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
	plt.ylim([0, 2000])
	plt.ylabel('Cost')
	plt.xlabel('Epochs')
	plt.tight_layout()
	plt.show()

else:	
	#載入權重
	load_w1=loadWeight1()
	nn.w1=load_w1.reshape(nn.w1.shape[0],nn.w1.shape[1])
	print("load_w1=",nn.w1)
	print("load_w1.shape=",nn.w1.shape)
	load_w2=loadWeight2()
	nn.w2=load_w2.reshape(nn.w2.shape[0],nn.w2.shape[1])
	print("load_w2=",nn.w2)
	print("load_w2.shape=",nn.w2.shape)


#預測測試樣本
y_test_pred = nn.predict2(X_test,nn.w1,nn.w2)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))


##My Predict testing +
##預測自己輸入的手寫數字圖
#自己手寫的20個數字
My_X =np.zeros((20,784), dtype=int) 
#自己手寫的20個數字對應的正確期望數字
My_Yd =np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9], dtype=int) 
img_num=[0]*20
img_res=[0]*20
#輸入20個手寫數字圖檔28x28=784 pixel，
Input_Numer=[0]*20
Input_Numer[0]="0_1.jpg"
Input_Numer[1]="1_1.jpg"
Input_Numer[2]="2_1.jpg"
Input_Numer[3]="3_1.jpg"
Input_Numer[4]="4_1.jpg"
Input_Numer[5]="5_1.jpg"
Input_Numer[6]="6_1.jpg"
Input_Numer[7]="7_1.jpg"
Input_Numer[8]="8_1.jpg"
Input_Numer[9]="9_1.jpg"
Input_Numer[10]="0_2.jpg"
Input_Numer[11]="1_2.jpg"
Input_Numer[12]="2_2.jpg"
Input_Numer[13]="3_2.jpg"
Input_Numer[14]="4_2.jpg"
Input_Numer[15]="5_2.jpg"
Input_Numer[16]="6_2.jpg"
Input_Numer[17]="7_2.jpg"
Input_Numer[18]="8_2.jpg"
Input_Numer[19]="9_2.jpg"

for i in range(20):  #read 20 digits picture
	img = cv2.imread(Input_Numer[i],0)    #Gray
	img_num[i]=img.copy()
	img=img.reshape(My_X.shape[1])
	My_X[i] =img.copy()


My_test_pred = nn.predict(My_X)
print("期望值：",My_Yd)
print("預測值：",My_test_pred)
acc = np.sum(My_Yd == My_test_pred, axis=0) / My_X.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(20):  
	img_res[i] = np.zeros((64,64,3), np.uint8)
	img_res[i][:,:]=[255,255,255]
	if (My_test_pred[i]%10)==(i%10):
		cv2.putText(img_res[i],str(My_test_pred[i]),(15,52), font, 2,(0,255,0),3,cv2.LINE_AA)
	else:
		cv2.putText(img_res[i],str(My_test_pred[i]),(15,52), font, 2,(255,0,0),3,cv2.LINE_AA)

Input_Numer_name = ['Input 0', 'Input 1','Input 2', 'Input 3','Input 4',\
					'Input 5','Input 6', 'Input 7','Input8', 'Input9',\
					'Input 0', 'Input 1','Input 2', 'Input 3','Input 4',\
					'Input 5','Input 6', 'Input 7','Input8', 'Input9',
					]
					
predict_Numer_name =['predict 0', 'predict 1','predict 2', 'predict 3','predict 4', \
					'predict 5','predict6 ', 'predict 7','predict 8', 'predict 9',\
					'predict 0', 'predict 1','predict 2', 'predict 3','predict 4', \
					'predict 5','predict6 ', 'predict 7','predict 8', 'predict 9',
					]
				
for i in range(20):
	if i<10:
		plt.subplot(4,10,i+1),plt.imshow(img_num[i],cmap = 'gray')
		plt.title(Input_Numer_name[i]), plt.xticks([]), plt.yticks([])
		plt.subplot(4,10,i+11),plt.imshow(img_res[i],cmap = 'gray')
		plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])
	else:
		plt.subplot(4,10,i+11),plt.imshow(img_num[i],cmap = 'gray')
		plt.title(Input_Numer_name[i]), plt.xticks([]), plt.yticks([])
		plt.subplot(4,10,i+21),plt.imshow(img_res[i],cmap = 'gray')
		plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])
		
plt.show()
##My Predict testing -
