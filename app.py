from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
import pickle


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	fileName = 'rf_model.pkl'
	with open(fileName , 'rb') as f:
		count_vec , rf_model  = pickle.load(f)
	if request.method == 'POST':
		message = request.form['message']
		my_prediction = rf_model.predict(count_vec.transform([message]))
	return  render_template('result.html' , prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)