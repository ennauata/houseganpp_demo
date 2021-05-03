from flask import Flask, jsonify, render_template, request, Response
from python._infer import run_model
import time
import json
from flask_cors import CORS
from waitress import serve

application = app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
	# receive post
	graph_str = request.data.decode('utf-8')
	graph_data = json.loads(graph_str)
	return Response(run_model(graph_data), mimetype='text/plain')
# serve(app, host='127.0.0.1', port=5000)
serve(app, host='0.0.0.0', port=80)
