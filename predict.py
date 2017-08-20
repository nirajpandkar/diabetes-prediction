from keras.models import model_from_json
import pandas
# Load JSON and create a model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

print "Loaded model"

dataset = pandas.read_csv("pima-indians-diabetes.data.1", delimiter=",")

array = dataset.values
print dataset.shape
X = array[:, 0:8]
y = array[:, 8]

loaded_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
scores = loaded_model.evaluate(X,y)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
