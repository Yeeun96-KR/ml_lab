from flask import Flask, render_template
import numpy as np
import pickle

#1. ML모델 메모리 로딩
with open('models/iris_model_svc.pkl', 'rb') as file:
    model = pickle.load(file)

# flask 객체생성
app = Flask(__name__)
# http://127.0.0.1:5000/
@app.route('/', methods=['GET', 'POST'])
def index():
    aaa = 'hello flask'
    bbb = 'static/flower1.jpg'
    print(aaa)
    return render_template('index.html', predict=aaa, img_path=bbb)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)