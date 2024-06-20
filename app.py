from flask import Flask, request, jsonify, render_template
from chat import get_response
import time

app = Flask(__name__)

@app.get('/')
def index_get():
    return render_template('base.html')

@app.post('/predict')
def predict():
    
    #wait code for 2 seconds
    time.sleep(2)
    text = request.get_json().get("message")
    #TODO check if the text is empty and valid
    response = get_response(text)
    return jsonify({'answer': response})

if __name__ == "__main__":
    app.run(debug=True)
