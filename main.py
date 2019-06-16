#import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
import keras
from scipy import misc
import pdb
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils.vis_utils import plot_model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras import applications
from keras.optimizers import Nadam
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.callbacks import CSVLogger
from keras.layers import BatchNormalization
#from model import MobileNetv2
#from mobilenet_v2 import MobileNetv2

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class dataSet(object):
	def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
		self.X_train = X_train
		self.X_val = X_val
		self.X_test = X_test
		self.y_train = y_train
		self.y_val = y_val
		self.y_test = y_test
		self.train_data = 0
		self.val_data = 0
	

class Config(object):
	def __init__(self, nc, ne, msl, bs, lr, dp, sv):
		self.num_classes = nc
		self.num_epochs = ne
		self.max_seq_len = msl
		self.batch_size = bs
		self.learning_rate = lr
		self.dropout = dp
		self.MAX_WIDTH = 48
		self.MAX_HEIGHT = 48
		self.class_names = ["begin","choose","connection","navigation","next","previous","start","stop","hello","web"]
		#self.class_names = ["stop navigation","excuse me","I am sorry","thank you","good bye","I love this game","nice to meet you","you are welcome","how are you","have a good time"]
		self.seen_validation = sv
		
	
		

class LipReader(object):
	def __init__(self, config):
		self.config = config	
		self.iteration = 0	
		
		#self.config.batch_size = np.shape(self.X_train)[0]
		#self.config.batch_size_val = np.shape(self.X_val)[0]

	def update_config(self,config):
		self.config = config
	

	def training_generator(self):
		while True:
			for i in range(int(np.shape(self.train_data)[0] / self.config.batch_size)):
				x = self.train_data[i * self.config.batch_size : (i + 1) * self.config.batch_size]
				y = self.y_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
				one_hot_labels_train = keras.utils.to_categorical(y, num_classes=self.config.num_classes)
				yield (x,one_hot_labels_train)


	def create_bottleneck_model(self, seen_validation):
		np.random.seed(0)
		if seen_validation is True:
			bottleneck_train_path = 'bottleneck_features_train_seen.npy'
			bottleneck_val_path = 'bottleneck_features_val_seen.npy'
			bottleneck_test_path = 'bottleneck_features_test_seen.npy'
			bottleneck_train_labels_path = 'bottleneck_features_train_labels_seen.npy'
		else:
			bottleneck_train_path = 'bottleneck_features_train_unseen.npy'
			bottleneck_val_path = 'bottleneck_features_val_unseen.npy'
			bottleneck_test_path = 'bottleneck_features_test_unseen.npy'
			bottleneck_train_labels_path = 'bottleneck_features_train_labels_unseen.npy'
	
		if not os.path.exists(bottleneck_train_path):
			#self.DataAugmentation()
			input_layer = keras.layers.Input(shape=(self.config.max_seq_len, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
			# build the VGG16 network
			#vgg_base = VGGFace(weights='vggface', include_top=False, input_shape=(self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
			vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
			vgg = Model(inputs=vgg_base.input, outputs=vgg_base.output)
			#vgg.trainable = False
			for layer in vgg.layers[:15]:
				layer.trainable = False
			x = TimeDistributed(vgg)(input_layer)
			bottleneck_model = Model(inputs=input_layer, outputs=x)
			if not os.path.exists(bottleneck_train_path):
				#bottleneck_features_train = bottleneck_model.predict_generator(self.training_generator(), steps=np.shape(self.X_train)[0] / self.config.batch_size)
				bottleneck_features_train = bottleneck_model.predict(self.X_train)
				np.save(bottleneck_train_path, bottleneck_features_train)
			if not os.path.exists(bottleneck_val_path):
				bottleneck_features_val = bottleneck_model.predict(self.X_val)
				np.save(bottleneck_val_path, bottleneck_features_val)
			if not os.path.exists(bottleneck_test_path):
				bottleneck_features_test = bottleneck_model.predict(self.X_test)
				np.save(bottleneck_test_path, bottleneck_features_test)
			if not os.path.exists(bottleneck_train_labels_path):
				np.save(bottleneck_train_labels_path, self.y_train)

		bottleneck_model.summary()
		plot_model(bottleneck_model, to_file='bottleneck_model_plot.png', show_shapes=True, show_layer_names=True)

	def create_model(self, seen_validation):
		np.random.seed(0)
		model = Sequential()
		model.add(TimeDistributed(keras.layers.core.Flatten(),input_shape=self.train_data.shape[1:]))
		lstm = keras.layers.recurrent.LSTM(256)
		model.add(keras.layers.wrappers.Bidirectional(lstm, merge_mode='concat', weights=None))
		#model.add(BatchNormalization())
		#model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

		model.add(keras.layers.core.Dropout(rate=self.config.dropout))
		model.add(keras.layers.core.Dense(10))
		model.add(keras.layers.core.Activation('softmax'))
		adam = keras.optimizers.Adam(lr=self.config.learning_rate)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
		one_hot_labels_val = keras.utils.to_categorical(self.y_val, num_classes=self.config.num_classes)
		one_hot_labels_test = keras.utils.to_categorical(self.y_test, num_classes=self.config.num_classes)
		one_hot_labels_train = keras.utils.to_categorical(self.train_data_label, num_classes=self.config.num_classes)
		print('Fitting the model...')
		self.config.class_names = np.array(self.config.class_names)
		self.iteration += 1
		if seen_validation is True:
			fileName = 'seen_results/csv/epoch_{0}.log'.format(self.iteration)
		else:
			fileName = 'unseen_results/csv/epoch_{0}.log'.format(self.iteration)

		csv_logger = keras.callbacks.CSVLogger(fileName, separator=',', append=True)
        #self.plot_confusion_matrix(self.y_test, y_pred,  title='Confusion matrix, without normalization')
		model.summary()
		plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
		history = model.fit(self.train_data, one_hot_labels_train, epochs=self.config.num_epochs, batch_size=self.config.batch_size,validation_data=(self.val_data, one_hot_labels_val),callbacks=[csv_logger])
		self.create_save_plots(history,model)
		self.evaluate_model(model,one_hot_labels_test,one_hot_labels_val)

	def DataAugmentation(self):
		#Generates 2 sets of augmented image from the given image and adds it to the training array
		if self.config.seen_validation is True:
			trainingDataSize = 1200
			testDataSize = 150
		else:
			trainingDataSize = 1300
			testDataSize = 100
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=1,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images
		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		print(np.shape(self.X_train)[0])
		#x = np.reshape(self.X_train,(22, 30, 30, 3))
		
		x_sequence_1 = []
		y_sequence_1 = []
		x_sequence_2 = []
		y_sequence_2 = []
		x_new_1 = []
		x_new_2 = []
		y_new_1 = []
		y_new_2 = []

		for i in range(trainingDataSize):

			x = self.X_train[i : (i + 1)]
			y = self.y_train[i : (i + 1)]
			x = np.reshape(x,(self.config.max_seq_len ,self.config.MAX_WIDTH  , self.config.MAX_HEIGHT , 3))
			#print ("x_train data")
			#print(self.X_train)
			one_hot_labels_train_ = keras.utils.to_categorical(y, num_classes=self.config.num_classes)
			y = np.array([y[0]] * self.config.max_seq_len )
			#y= y * 22
			one_hot_labels_train = keras.utils.to_categorical(y, num_classes=self.config.num_classes)

			x = np.reshape(x,(self.config.max_seq_len ,self.config.MAX_WIDTH  , self.config.MAX_HEIGHT , 3))
			y_sequence_1.append(y)
			y_sequence_2.append(y)
			for k in range(self.config.max_seq_len ):


				x_test = x[k : (k + 1)]
				y_test = y[k : (k + 1)]
				#x_show = np.reshape(x_test,(self.config.MAX_WIDTH  , self.config.MAX_HEIGHT , 3))
				#x_show = x_show.astype(np.uint8)
				#print ("x_test data")
				#print(x_test)
				#cv2.imshow('image',x_show)
				#cv2.waitKey(0)

				j = 0
				maxval = int(np.amax(x_test))
				if maxval is 0: 
					x_sequence_1.append(x_test)
					x_sequence_2.append(x_test)

				else:
					for x_temp, y_temp in datagen.flow(x_test, one_hot_labels_train_,batch_size=2, save_to_dir='images', save_prefix='aug', save_format='png'):
					#for x_temp, y_temp in datagen.flow(x_test, one_hot_labels_train_):
						#x_temp = np.reshape(x_temp, (1,self.config.max_seq_len , self.config.MAX_WIDTH ,self.config.MAX_HEIGHT , 3))
						if j is 0:
							x_sequence_1.append(x_temp)
							#y_sequence_1.append(y_test)
						if j is 1 :
							x_sequence_2.append(x_temp)
							#y_sequence_2.append(y_test)
						j += 1
						if j is 2 :
							break
			print(i)
			if i >= trainingDataSize:
				break
		x_sequence_1 = np.reshape(x_sequence_1,(trainingDataSize,self.config.max_seq_len ,self.config.MAX_WIDTH,self.config.MAX_WIDTH,3))
		x_sequence_2 = np.reshape(x_sequence_2,(trainingDataSize,self.config.max_seq_len ,self.config.MAX_WIDTH,self.config.MAX_WIDTH,3))
		#y_sequence_1 = np.reshape(y_sequence_1, (1200,))
		#y_sequence_2 = np.reshape(y_sequence_2, (1200,))
		x_new_1= np.append(self.X_train,x_sequence_1)
		x_new_2= np.append(x_new_1, x_sequence_2)
		x_new_2 = np.reshape(x_new_2, (trainingDataSize * 3,self.config.max_seq_len ,self.config.MAX_WIDTH,self.config.MAX_WIDTH,3))
		y_new_1 = np.append(self.y_train,self.y_train)
		y_new_2 = np.append(y_new_1,self.y_train)
		self.X_train = x_new_2
		self.y_train = y_new_2

	def evaluate_model(self,model,one_hot_labels_test,one_hot_labels_val):
		print('Evaluating the model...')
		score = model.evaluate(self.val_data, one_hot_labels_val, batch_size=self.config.batch_size)
		print('Finished training, with the following val score:')
		print(score)
		print('Evaluating the model...')
		score = model.evaluate(self.test_data, one_hot_labels_test, batch_size=self.config.batch_size)
		print('Finished training, with the following val score:')
		print(score)

	def create_save_plots(self,history,model):
		self.create_plots(history)
		self.plot_and_save_cm(model)

	def plot_and_save_cm(self,model):
		if self.config.seen_validation is True:
			fileName = 'seen_results/plots/conf_matrix_test_{0}.png'.format(self.iteration)
		else:
			fileName = 'unseen_results/plots/conf_matrix_test_{0}.png'.format(self.iteration)
		y_pred = model.predict_classes(self.test_data, verbose=1)
		self.plot_confusion_matrix(self.y_test, y_pred, classes=self.config.class_names,fileName=fileName)

		if self.config.seen_validation is True:
			fileName = 'seen_results/plots/conf_matrix_val_{0}.png'.format(self.iteration)
		else:
			fileName = 'unseen_results/plots/conf_matrix_val_{0}.png'.format(self.iteration)
		y_pred = model.predict_classes(self.val_data, verbose=1)
		self.plot_confusion_matrix(self.y_val, y_pred, classes=self.config.class_names,fileName=fileName)

	def create_plots(self, history):
		if not os.path.exists('plots'):
			os.mkdir('plots')
		# summarize history for accuracy
		print("create_plots {0}".format(self))
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		if self.config.seen_validation is True:
			fileName = 'seen_results/plots/acc_plot_{0}.png'.format(self.iteration)
		else:
			fileName = 'unseen_results/plots/acc_plot_{0}.png'.format(self.iteration)

		plt.savefig(fileName)
		plt.clf()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		if self.config.seen_validation is True:
			fileName = 'seen_results/plots/loss_plot_{0}.png'.format(self.iteration)
		else:
			fileName = 'unseen_results/plots/loss_plot_{0}.png'.format(self.iteration)

		plt.savefig(fileName)
		plt.clf()


	def plot_confusion_matrix(self,y_true, y_pred, classes,fileName,
							normalize=False,
							title=None,
							cmap=plt.cm.Blues,
							):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		if not title:
			if normalize:
				title = 'Normalized confusion matrix'
			else:
				title = 'Confusion matrix, without normalization'

		# Compute confusion matrix
		cm = confusion_matrix(y_true, y_pred)
		# Only use the labels that appear in the data
		classes = classes[unique_labels(y_true, y_pred)]
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		print(cm)

		fig, ax = plt.subplots()
		im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
		ax.figure.colorbar(im, ax=ax)
		# We want to show all ticks...
		ax.set(xticks=np.arange(cm.shape[1]),
			yticks=np.arange(cm.shape[0]),
			# ... and label them with the respective list entries
			xticklabels=classes, yticklabels=classes,
			title=title,
			ylabel='True label',
			xlabel='Predicted label')

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
				rotation_mode="anchor")

		# Loop over data dimensions and create text annotations.
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				ax.text(j, i, format(cm[i, j], fmt),
						ha="center", va="center",
						color="white" if cm[i, j] > thresh else "black")
		fig.tight_layout()
		
		plt.savefig(fileName)
		plt.clf()
		return ax


	def load_data(self, seen_validation):
		if seen_validation is True:
			bottleneck_train_path = 'bottleneck_features_train_seen.npy'
			bottleneck_val_path = 'bottleneck_features_val_seen.npy'
			bottleneck_test_path = 'bottleneck_features_test_seen.npy'
			bottleneck_train_labels_path = 'bottleneck_features_train_labels_seen.npy'
		else:
			bottleneck_train_path = 'bottleneck_features_train_unseen.npy'
			bottleneck_val_path = 'bottleneck_features_val_unseen.npy'
			bottleneck_test_path = 'bottleneck_features_test_unseen.npy'
			bottleneck_train_labels_path = 'bottleneck_features_train_labels_unseen.npy'

		self.train_data = np.load(bottleneck_train_path)
		self.val_data = np.load(bottleneck_val_path)	
		self.test_data = np.load(bottleneck_test_path)
		self.train_data_label= np.load(bottleneck_train_labels_path)

	def load_bottleneck_data(self, seen_validation):

		self.limited_set = False

		data_dir = 'data'
		if seen_validation:
			data_dir = 'data_seen'

		print(data_dir)
		if os.path.exists('../' + data_dir):
			print('loading saved data...')
			self.X_train = np.load('../' +  data_dir + '/X_train.npy')
			self.y_train = np.load('../'+ data_dir +'/y_train.npy')

			self.X_val = np.load('../'+ data_dir +'/X_val.npy')
			self.y_val = np.load('../'+data_dir+'/y_val.npy')

			self.X_test = np.load('../'+data_dir+'/X_test.npy')
			self.y_test = np.load('../'+data_dir+'/y_test.npy')
			print('Read data arrays from disk.npy')
			
			#self.X_test = np.reshape(self.X_test, (np.shape(self.X_test)[0], -1, 480*640*3))


		else:

			people_full_set = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
			people_limited = ['F01']
			
			#removed 'phrases' temporarily from data types
			data_types = ['words']#, 'words_jitter']#, 'words_flip_xaxis']
			
			folder_enum = ['01','02','03','04','05','06','07','08','09','10']
			#folder_enum = ['01','02','03','04']

			UNSEEN_VALIDATION_SPLIT = ['F05']
			UNSEEN_TEST_SPLIT = ['F06']

			SEEN_VALIDATION_SPLIT = ['02']
			SEEN_TEST_SPLIT = ['01']

			self.X_train = []
			self.y_train = []

			self.X_val = []
			self.y_val = []

			self.X_test = []
			self.y_test = [] 
			if self.limited_set:
				people = people_limited
			else: 
				people = people_full_set
			directory = 'output4848_'
			
			for person_id in people:
				for data_type in data_types: 
					for word_index, word in enumerate(folder_enum):
						for iteration in folder_enum:
							path = os.path.join(person_id, 'words', word, iteration,directory)
							filelist = sorted(os.listdir(path + '/'))
							#filelist = os.listdir(path + '/')
							sequence = []
							for img_name in filelist:
								if img_name.startswith('face'):
									image = misc.imread(path + '/' + img_name)
									#image = cv2.resize(image, (self.config.MAX_WIDTH, self.config.MAX_HEIGHT))
									#image = image[:self.config.MAX_WIDTH,:self.config.MAX_HEIGHT,...]
									#image = np.reshape(image, self.config.MAX_WIDTH*self.config.MAX_HEIGHT*3)
									sequence.append(image)
									print("read: " + path + '/' + img_name)
							pad_array = [np.zeros((self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))]
							sequence.extend(pad_array * (self.config.max_seq_len - len(sequence)))
							sequence = np.stack(sequence, axis=0)
							
							if seen_validation == False:
								if person_id in UNSEEN_TEST_SPLIT:
									self.X_test.append(sequence)
									self.y_test.append(word_index)
								elif person_id in UNSEEN_VALIDATION_SPLIT:
									self.X_val.append(sequence)
									self.y_val.append(word_index)
								else:
									self.X_train.append(sequence)
									self.y_train.append(word_index)
							else:
								if iteration in SEEN_TEST_SPLIT:
									self.X_test.append(sequence)
									self.y_test.append(word_index)
								elif iteration in SEEN_VALIDATION_SPLIT:
									self.X_val.append(sequence)
									self.y_val.append(word_index)
								else:
									self.X_train.append(sequence)
									self.y_train.append(word_index)

				print('Finished reading images for person ' + person_id)
			
			print('Finished reading images.')
			print(np.shape(self.X_train))
			self.X_train = np.stack(self.X_train, axis=0)	
			self.X_val = np.stack(self.X_val, axis=0)
			self.X_test = np.stack(self.X_test, axis=0)
			print('Finished stacking the data into the right dimensions. About to start saving to disk...')		
			os.mkdir('../' + data_dir)
			np.save('../'+data_dir+'/X_train', self.X_train)
			np.save('../'+data_dir+'/y_train', np.array(self.y_train))
			np.save('../'+data_dir+'/X_val', self.X_val)
			np.save('../'+data_dir+'/y_val', np.array(self.y_val))
			np.save('../'+data_dir+'/X_test', self.X_test)
			np.save('../'+data_dir+'/y_test', np.array(self.y_test))
			print('Finished saving all data to disk.')

		print('X_train shape: ', np.shape(self.X_train))
		print('y_train shape: ', np.shape(self.y_train))

		print('X_val shape: ', np.shape(self.X_val))
		print('y_val shape: ', np.shape(self.y_val))

		print('X_test shape: ', np.shape(self.X_test))
		print('y_test shape: ', np.shape(self.y_test))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Lip reading model')
	parser.add_argument('--seen_validation', dest='seen_validation', action='store_true')
	#parser.add_argument('--unseen_validation', dest='seen_validation', action='store_false')
	
	for i in range(2):
		if i is 0:
			parser.set_defaults(seen_validation=True)
		else:
			parser.set_defaults(seen_validation=False)
		ARGS = parser.parse_args()
		print("Seen validation: %r" % (ARGS.seen_validation))
		config = Config(10, 0, 22, 0, 0, 0, ARGS.seen_validation)
		lipReader = LipReader(config)
		lipReader.load_bottleneck_data(ARGS.seen_validation)
		lipReader.create_bottleneck_model(ARGS.seen_validation)
		lipReader.load_data(ARGS.seen_validation)
		
		num_epochs = [35]#10
		learning_rates = [0.0001, 0.0002, 0.0005]
		#learning_rates = [0.0005]
		batch_size = [64]
		dropout_ = [ 0.1, 0.3,0.5]
		#dropout_ = [ 0.4]
		for ne in num_epochs:
			for bs in batch_size: 
				for lr in learning_rates:
					for dp in dropout_:
						print("Epochs: {0}    Batch Size:{1}  Learning Rate: {2} Dropout {3}".format(ne, bs, lr, dp))
						config = Config(10, ne, 22, bs, lr, dp,ARGS.seen_validation)
						lipReader.update_config(config)
						lipReader.create_model(ARGS.seen_validation)
