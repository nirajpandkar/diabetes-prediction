from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import optimizers
import time
from sklearn import model_selection

import numpy
import pandas
seed = 7
numpy.random.seed(seed)

# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# dataset = numpy.loadtxt("pima-indians-diabetes.data", delimiter=",")


names = ['no_pregnant', 'pla_glu_concentration', 'blood_pressure(mmHg)', 'skin_fold_thickness(mm)', 'serum_insulin(U/ml)', 'bmi', 'diab_pedigree_fun', 'age', 'class']

dataset = pandas.read_csv('pima-indians-diabetes.data', delimiter=',', names=names)

arrays = dataset.values

# print dataset.shape
# print dataset.head(10)
# print dataset.describe()
# print dataset.groupby('no_pregnant').size()

X=arrays[:,0:8]
Y=arrays[:,8]

validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)




#Create model
model = Sequential()
model.add(Dense(32, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set log directory for tensorboard
tensorboard = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True, write_images=True)

# Fit the model
model.fit(X_train, Y_train, nb_epoch=700, batch_size=10, callbacks=[tensorboard])

# scores = model.evaluate(X_validation,Y_validation)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights("model.h5")
print "Saved model"
