import flask
from flask import Flask, jsonify, request, render_template
import sys
sys.path.append("..")
import skipthoughts 

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	input_query = request.get_json(silent=True, force=True)['input']




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)