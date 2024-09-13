import streamlit as st
import numpy as np
import pickle  # pickle로 변경
from PIL import Image

# 모델 불러오기 (pickle 사용)
with open('models/iris_model_svc.pkl', 'rb') as file:
    model = pickle.load(file)

# 레이블에 해당하는 이미지 불러오기 함수
def get_iris_image(label):
    if label == 0:
        return Image.open('static/setosa.jpg')  # Setosa 이미지
    elif label == 1:
        return Image.open('static/versicolor.jpg')  # Versicolor 이미지
    else:
        return Image.open('static/virginica.png')  # Virginica 이미지

# 기본 이미지 불러오기 (예측 전 보여줄 이미지)
default_image = Image.open('static/flower1.jpg')

# 사용자 입력 받기
st.title('Iris Species Prediction')

sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# 기본 이미지 출력
st.image(default_image, caption="Input the flower details to predict the species", use_column_width=True)

# 예측 버튼
if st.button('Predict'):
    # 입력 데이터를 모델에 맞게 변환
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # 예측 수행
    prediction = model.predict(input_data)
    label = prediction[0]
    
    # 레이블과 이미지 출력
    if label == 0:
        st.write('Predicted Species: Setosa')
    elif label == 1:
        st.write('Predicted Species: Versicolor')
    else:
        st.write('Predicted Species: Virginica')
    
    # 해당 레이블 이미지 출력
    st.image(get_iris_image(label), caption=f'Iris {label}', use_column_width=True)
