from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import os
# instanciate a Flask application, __name__ will reference this file, embedding it in a flask server
app = Flask(__name__)

# set up flags for upload security checks
app.config['UPLOAD_EXTENSIONS'] = ['jpeg', 'jpg', 'png', 'gif']
app.config['MAX_CONTENT_LENGTH'] = 1024*1024

# handle initial page
@app.route('/', methods = ['GET','POST'])
def index():

    if request.method == 'POST':
        # loop through uploaded files
        for idx, file_to_upload in enumerate(request.files.getlist('file')):
            filename = file_to_upload.filename
            # check file not empty
            if filename != '':
                # check file extension
                if filename.split(".")[-1] not in app.config['UPLOAD_EXTENSIONS']:
                    return "ERROR: unsupported file extension: {}. "\
                           "Supported extensions: {}".format(filename.split(".")[-1],
                                                             app.config['UPLOAD_EXTENSIONS'])
                # save file to cwd
                savepath = os.path.join(os.getcwd(), "temp")
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                file_to_upload.save(os.path.join(savepath, "sample{}.png".format(idx)))
            # redirect to initial page    
            return redirect(url_for('index'))
    else:
        # render initial page template to not immediately 404
        return render_template('homepage.html')

# predict on the uploaded files
@app.route('/predict')
def predict():
    # 1 transform uploaded images to tensors
    tensor_images = []
    for file in os.listdir("./temp"):
        tensor_images.append(transorm_image(file))




if __name__ == "__main__":
    app.run(debug=True)