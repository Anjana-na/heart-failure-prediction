import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


model = pickle.load(open("heart_pickle", "rb"))
app = Flask(__name__)



@app.route('/')
def man():
	return render_template("home.html")



@app.route('/predict',methods=['POST'])
def home():
	data1 = request.form["age"]
	data2 = request.form["anaimea"]
	data3 = request.form["creatinine phosphokinase"]
	data4 = request.form["diabetes"]
	data5 = request.form["ejection fraction"]
	data6 = request.form["high blood pressure"]
	data7 = request.form["platelets"]
	data8 = request.form["serum creatinine"]
	data9 = request.form["serum sodium"]
	data10 = request.form["sex"]
	data11 = request.form["smoking"]
	data12 = request.form["time"]
	arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12]])
	pred = model.predict(arr)
	return render_template("after.html", data=pred)






if __name__ == "__main__":
	app.run(debug=True)

	



	