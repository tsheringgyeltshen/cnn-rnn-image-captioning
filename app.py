from flask import Flask, render_template, request
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse



app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route("/", methods=['GET','POST'])
def get_output():
    if request.method=='POST':
        img = request.files['my_image']

        img_path="./static/images/" + img.filename
        img.save(img_path)

        def extract_features(filename, model):
            try:
                image = Image.open(filename)
                
            except:
                print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
            image = image.resize((299,299))
            image = np.array(image)
            # for images that has 4 channels, we convert them into 3 channels
            if image.shape[2] == 4: 
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            return feature

        def word_for_id(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None


        def generate_desc(model, tokenizer, photo, max_length):
            in_text = ''
            for i in range(max_length):
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=max_length)
                pred = model.predict([photo,sequence], verbose=0)
                pred = np.argmax(pred)
                word = word_for_id(pred, tokenizer)
                if word is None:
                    break
                in_text += ' ' + word
                if word == 'end':
                    break
            return in_text


        # path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
        max_length = 32
        tokenizer = load(open("tokenizer.p","rb"))
        model = load_model('model_30.h5')
        xception_model = Xception(include_top=False, pooling="avg")

        photo = extract_features(img_path, xception_model)
        img = Image.open(img_path)

        description = generate_desc(model, tokenizer, photo, max_length)
        print("\n\n")
        print(description)
        plt.imshow(img)


    return render_template("index.html", p = description)






