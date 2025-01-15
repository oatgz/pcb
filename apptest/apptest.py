from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Assuming the image is located in the 'static' folder
    image_path = os.path.join('static', 'test_image.jpg')  # Replace with your image filename
    return render_template('index.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
