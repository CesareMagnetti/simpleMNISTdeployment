from flask import Flask, render_template, url_for, request, redirect, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import os
from torch_utils import transformImage, getPrediction
# instanciate a Flask application, __name__ will reference this file, embedding it in a flask server
app = Flask(__name__)

# set up flags for upload security checks
app.config['UPLOAD_EXTENSIONS'] = ['jpeg', 'jpg', 'png', 'gif']
app.config['UPLOAD_PATH'] = os.path.join(os.getcwd(), "uploads")

# get torch utils to transform images and get predictions
transform_image = transformImage()
get_prediction = getPrediction()

# handle initial page
@app.route('/')
def index():
    try:
        files = os.listdir(app.config['UPLOAD_PATH'])
        return render_template('homepage.html', files = files)
    except:
        return render_template('homepage.html') # if no file has been uploaded yet


# function to upload files
@app.route('/', methods = ['POST'])
def upload_files():
    # loop through uploaded files
    for file_to_upload in request.files.getlist('file'):
        filename = file_to_upload.filename
        # check file not empty
        if filename != '':
            # check file extension
            if filename.split(".")[-1] not in app.config['UPLOAD_EXTENSIONS']:
                return "ERROR: unsupported file extension: {}. "\
                        "Supported extensions: {}".format(filename.split(".")[-1],
                                                            app.config['UPLOAD_EXTENSIONS'])

            # save file(s)
            if not os.path.exists(app.config['UPLOAD_PATH']):
                os.mkdir(app.config['UPLOAD_PATH'])
            file_to_upload.save(os.path.join(app.config['UPLOAD_PATH'],filename))
  
    # redirect to initial page    
    return redirect('/')

# function to retrieve a particular file
@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

# remove uploaded file
@app.route('/delete/<filename>')
def delete(filename):
    os.remove(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('index'))

# predict on the uploaded files
@app.route('/predict')
def predict():
    
    out = []
    files = os.listdir(app.config['UPLOAD_PATH'])
    for filename in files:
        # 1 transform file to tensor
        try:
            with(open(os.path.join(app.config['UPLOAD_PATH'],
                                filename), 'rb')) as f:
                tensor_image = transform_image(f.read())
        except:
            return "Something went wrong while preprocessing image: {}".format(filename)
    
        # 2 get prediction
        try:
            pred = get_prediction(tensor_image)
        except:
            return "Something went wrong while making predictions on image: {}".format(filename)
    
        # 3 append to out
        out.append({'filename': filename, 'class': pred})

    # 4 return jsonify of the dict
    return render_template('predict.html', files=out)


if __name__ == "__main__":
    app.run(debug=True)