from flask import Flask, render_template, request
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
        
    # 사용자 입력 데이터를 받아서 예측수행
    ## 클라이언트에서 넘어온 요청이 request == POST
    if request.method == "POST":
        ## 4개의 데이터를 value를 뽑고
        # print(request.form['sl'])
        # print(request.form['sw'])
        # print(request.form['pl'])
        # print(request.form['pw'])

        sl = request.form['sl']
        sw = request.form['sw']
        pl = request.form['pl']
        pw = request.form['pw']

        ## numpy 2D로 만들고
        input_data = [[sl, sw, pl, pw]]

        ## 모델 예측수행 -> 결과 index.html에 랜더링
        predict = model.predict(input_data)
        result = predict[0]

        ## 이미지 가져오기
        if result == 0:
            img_path = 'static/setosa.jpg'
            predict = 'setosa'
        elif result == 1:
            img_path = 'static/versicolor.jpg'
            predict = 'versicolor'
        else:
            img_path = 'static/virginica.png'
            predict = 'virginica'

        return render_template('index.html', predict=predict, img_path=img_path)
    # 데이터 입력 페이지 처리(get)
    return render_template('index.html', img_path = 'static/flower1.jpg')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)