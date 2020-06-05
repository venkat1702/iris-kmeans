from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict',methods=["POST"])
def analyze():
	dataset=pd.read_csv("iris.csv")
	x = dataset.iloc[:,0:4 ].values
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
	y_kmeans = kmeans.fit_predict(x)

	if request.method == 'POST':
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']

		# Clean the data by convert from unicode to float 
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]
		ex1 = np.array(clean_data).reshape(1,-1)
		my_prediction = kmeans.predict(ex1)
		result=float(my_prediction)
	return render_template('index.html', prediction=result,petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length)


if __name__ == '__main__':
	app.run(debug=True)