# simple MNIST classifier deployment with Flask and PyTorch

Small project to deploy a pytorch model on heroku, potentially allowing for user inputs.<br> 
User interface will be minimal (if any) as this is not an HTML/css project but more a project to learn about deploying ML models as web applications.<br><br>


## Usage

1. Head over to: https://cesare-mnist-simple-app.herokuapp.com/ (may take a couple of seconds to load)
2. The homepage should look like the following:
<img src=“https://gitlab.com/cesare.magnetti/pytorchdeployment/-/blob/master/readme_images/homepage.png”>
3. Hoover over ```choose files``` and upload an image (note that a simple CNN model was trained on MNIST, don't expect it to have superpowers)
4. submit the image and click on ```get predictions``` to view the model's output, the final page should look like the following:
<img src=“https://gitlab.com/cesare.magnetti/pytorchdeployment/-/blob/master/readme_images/prediction.png”>

## Contributors
@cesare.magnetti

