import json
import unittest
import responses
import numpy as np
import cv2

from classifyc.app import flask_app as app


class TestInterface(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
    def tearDown(self):
        pass

    def test_health(self):
        r = self.client.get('/classifyc/health')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(r.data)['code'], 0)

    def test_invalid_args(self):
        r = self.client.post('/classifyc/api/predict',
                             json={'abc': 'cba'})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(json.loads(r.data)['code'], 400)

    def test_invalid_url(self):
        r = self.client.post('/classifyc/api/predict',
                             json={'url': 'http://'})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(json.loads(r.data)['code'], 400)

    @responses.activate
    def test_predict(self):
        image = np.zeros([400, 400, 3], np.uint8)
        _, img = cv2.imencode('.jpg', image)
        responses.add(
            responses.GET, 'http://example.org/images/blank.jpg',
            body=img.tobytes(), status=200,
            content_type='image/jpeg',
            stream=True
        )
        r = self.client.post('/classifyc/api/predict', json={
            'url': 'http://example.org/images/blank.jpg'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(r.data)['code'], 0)
        self.assertEqual(json.loads(r.data)['data'], '未检测到钱币')

    def test_inference(self):
        import os
        with open(os.path.dirname(__file__) + '/sztb.jpg', 'rb') as f:
            image = f.read()
        from io import BytesIO
        data = dict(
            img=(BytesIO(image), "img"),
        )
        r = self.client.post('/classifyc/api/inference', data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(r.data)['code'], 0)
        self.assertEqual(json.loads(r.data)['data'], '顺治通宝')

    def test_inference_get_data(self):
        import os
        with open(os.path.dirname(__file__) + '/sztb.jpg', 'rb') as f:
            image = f.read()
        from io import BytesIO
        r = self.client.post('/classifyc/api/inference', data=BytesIO(image))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(r.data)['code'], 0)
        self.assertEqual(json.loads(r.data)['data'], '顺治通宝')
