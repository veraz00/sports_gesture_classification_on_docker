import os 
import sys 
import pytest 


sys.path.insert(0, './')
sys.path.insert(0, '../')

from api import app 

@pytest.fixture
def client():
    app.config.update({"TESTING": True})

    with app.test_client() as client:
        yield client

def test_failure(client):
    response = client.get('/null')
    assert response.status_code == 404 


def test_upload(client):
    reponse = client.get('/')
    assert reponse.status_code == 200


SMALLEST_JPEG_B64 = """\
/9j/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8Q
EBEQCgwSExIQEw8QEBD/yQALCAABAAEBAREA/8wABgAQEAX/2gAIAQEAAD8A0s8g/9k=
"""
import werkzeug
import io
import base64  
import json 
from bs4 import BeautifulSoup 


def test_predict(client):
    fake_image = werkzeug.datastructures.FileStorage(
                stream=io.BytesIO(base64.b64decode(SMALLEST_JPEG_B64)),
                filename="example image.jpg",
                content_type="image/jpg",
            )
    response = client.post('/', data = {'image':fake_image}, content_type = 'multipart/form-data')
    assert response.status_code == 200
    assert response.content_type == 'text/html; charset=utf-8'
    soup = BeautifulSoup(response.data.decode('utf-8'), 'html.parser')
    prediction_text = soup.find('h3').text

    # print(prediction_text, 'prediction_text')
    parts = prediction_text.split()
    if len(parts) > 6:
        temp = ''
        for i in range(1, len(parts)-4):
            temp += parts.pop(1)
            temp += ' '
        
    parts.insert(1, temp)         
    # Initialize an empty dictionary
    result_dict = {}
    # # # print('parts', parts)
    # Iterate through the parts and extract the key-value pairs
    for i in range(0, len(parts), 2):
        key = parts[i].strip(':')
        value = parts[i + 1]
        result_dict[key] = value

    pred_index = int(result_dict['Label'])
    confidence = float(result_dict['Confidence'])
    assert pred_index < 100  # replace with your expected classes
    assert confidence >= 0.0 and confidence <= 1.0
