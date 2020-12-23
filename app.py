from flask import Flask, url_for, jsonify, redirect

app = Flask(__name__)

@app.route('/app/')
def index():
    return "hello world"

if __name__ == "__main__":
    app.run(debug = True)