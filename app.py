from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
# instanciate a Flask application, __name__ will reference this file, embedding it in a flask server
app = Flask(__name__)

# create route so that when we browse to the URL we dont error 404
# we pass in route() the URL string of where we want to route the app,
# for now we just rout to the main directory
@app.route('/', methods=['POST', 'GET'])
# define function for the abouve route
def predict():
    if request.method == 'POST':

        # 1 get image bytes
        # 2 transform image bytes to tensor
        # 3 make prediction
        # 4 return json file
        pass
    else:
        # render initial page template
        return render_template('homepage.html')

if __name__ == "__main__":
    app.run(debug=True)