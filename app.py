from flask import Flask, render_template, url_for, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from torch_utils import transformImage, getPrediction
# instanciate a Flask application, __name__ will reference this file, embedding it in a flask server
app = Flask(__name__)

# set up flags for upload security checks
app.config['UPLOAD_EXTENSIONS'] = ['jpeg', 'jpg', 'png', 'gif']
app.config['MAX_CONTENT_LENGTH'] = 1024*1024

# get torch utils to transform images and get predictions
transform_image = transformImage()
get_prediction = getPrediction()

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
    try: 
        for file in os.listdir("./temp"):
            with(open("./temp/" + file, 'rb')) as f:
                tensor_images.append(transform_image(f.read()))
            os.remove("./temp/" + file) # file not needed to be stored anymore
            os.rmdir("./temp")
    except:
        return "Something went wrong while preprocessing the uploaded images..."
    
    # 2 get predictions
    predictions = []
    try:
        for image in tensor_images:
            predictions.append(get_prediction(image))
    except:
        return "Something went wrong while making predictions on the uploaded images..."
    
    # 3 instanciate a dict to return
    ret = {'images': [], 'class': []}
    for im, pred in zip(tensor_images, predictions):
        ret['images'].append(im.detach().cpu().numpy().squeeze())
        ret['class'].append(pred)

    # 4 return jsonify of the dict
    return render_template('predict.html', ret=ret)


if __name__ == "__main__":
    app.run(debug=True)