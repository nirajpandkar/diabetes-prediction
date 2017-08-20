from keras.models import model_from_json
from sklearn import model_selection
import pandas
# Load JSON and create a model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

print "Loaded model"

dataset = pandas.read_csv("pima-indians-diabetes.data", delimiter=",")

array = dataset.values
X = array[:, 0:8]
Y = array[:, 8]

seed = 7
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


loaded_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
scores = loaded_model.evaluate(X_validation, Y_validation)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
