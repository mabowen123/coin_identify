import hashlib
import requests
from urllib.parse import urljoin

class Skylake:
    def __init__(self, env, headers_ut=""):
        if env == "test":
            LINK_HOST = "https://skt.weipaitang.com"
        elif env == "pre":
            LINK_HOST = "https://canary-sk.weipaitang.com"
        elif env == "prod":
            LINK_HOST = "http://sk.wptqc.com"
        else:
            raise CdnParameterException(10000, message="参数错误,请指定当前环境,{test|pre|prod}")

        self.LINK_HOST = LINK_HOST
        self.upload_url = urljoin(self.LINK_HOST, "/api/v1/links")

        if not headers_ut:
            headers_ut = "eyJhbGciOiJIUzI1NiIsImtpZCI6InVzZXJ8MjAyMC0wOS0xNSIsInR5cCI6IkpXVCJ9.eyJvcGVuaWQiOiJvR2RkY3M2WGVrcy1oUjlteUVlYy1za3lsYWtlIiwicGxhdGZvcm1JZCI6MCwiZXhwIjoxNjAzMTcwNzc3LCJpYXQiOjE2MDI1NjU5NzcsImlzcyI6IndwdCIsIm5iZiI6MTYwMjU2NTk3Nywic3ViIjoiYmFzZSJ9.KbN4fdMxhDiWMAm-iWdJToTE55ft4Yk90HoMTWYJ-P0"
        self.headers = {"ut": headers_ut}

    def image_link_upload(self, link_url: str, source="server", business_name="", scene_name=""):
        """
        link_url: 可直接下载的图片链接
        """
        if not source:
            raise CdnParameterException(10000, message='参数错误,source为空')
        if not business_name:
            raise CdnParameterException(10000, message='参数错误,business_name为空')
        if not scene_name:
            raise CdnParameterException(10000, message='参数错误,scene_name为空')

        data = {
            "source": source,
            "businessName": business_name,
            "sceneName": scene_name,
            "mediaType": "image",
            "link": link_url
        }

        r = requests.post(self.upload_url, json=data, headers=self.headers)
        return r.json()

    def image_binary_upload(self, content: bytes, source="server", business_name="", scene_name=""):
        # 通过响应上传
        if not source:
            raise CdnParameterException(10000, message='参数错误,source为空')
        if not business_name:
            raise CdnParameterException(10000, message='参数错误,business_name为空')
        if not scene_name:
            raise CdnParameterException(10000, message='参数错误,scene_name为空')

        md = hashlib.md5()
        md.update(content)
        _binary = md.hexdigest()
        params = {
            "source": source,
            "businessName": business_name,
            "sceneName": scene_name,
            "mediaType": "image",
            "md5": _binary,
        }

        files = {'file': content}
        url = urljoin(self.LINK_HOST, "/api/v1/binary")
        r = requests.post(url, data=params, files=files, headers=self.headers)
        return r.json()

    def file_upload(self, abs_file_path: str, source="server", business_name="", scene_name=""):
        # 通过文件绝对路径上传
        if not source:
            raise CdnParameterException(10000, message='参数错误,source为空')
        if not business_name:
            raise CdnParameterException(10000, message='参数错误,business_name为空')
        if not scene_name:
            raise CdnParameterException(10000, message='参数错误,scene_name为空')
        if "." not in abs_file_path:
            raise CdnParameterException(10000, message='参数错误,abs_file_path请检查文件类型')
        file_type = abs_file_path.split(".")[-1]

        params = {
            "source": source,
            "businessName": business_name,
            "sceneName": scene_name,
            "mediaType": 'other',
            "fileSuffix": file_type,
            "md5": md5_file(abs_file_path),
        }

        content = open(abs_file_path, 'rb')
        files = {'file': content}
        url = urljoin(self.LINK_HOST, "/api/v1/binary")
        r = requests.post(url, data=params, files=files, headers=self.headers)
        content.close()

        return r.json()

class CdnParameterException(Exception):
    """参数错误"""
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super(CdnParameterException, self).__init__()

def md5_file(abs_file_path):
    fp = open(abs_file_path, 'rb')
    md = hashlib.md5()
    md.update(fp.read())
    fp.close()
    return md.hexdigest()


def test_download_video():
    url = 'https://tbm-auth.alicdn.com/d12952677edc7846/Ul10vxNlbwJO2L2ZchN/jUaAKzu0GddkSWeg9uq_267425464861_hd_append.mp4?auth_key=1622011085-0-0-b46153b1c76ee9142ccfe7df187c2e13'
    r = requests.get(url)
    with open('天猫.mp4', 'wb') as w:
        w.write(r.content)

def test_link_upload():
    test_image_url = 'https://img.alicdn.com/imgextra/https://img.alicdn.com/imgextra/i1/2258857036/O1CN01DiWCVi21qXLbYkTdK_!!2258857036.jpg'
    cu = Skylake(env="test")
    result = cu.image_link_upload(test_image_url, business_name="ssdspider", scene_name="tupian")
    print(result)

def test_binary_upload():
    """二进制图片上传, test_image_url需要绝对路径"""
    test_image_url = './skylake_test.jpg'
    fp = open(test_image_url, 'rb')
    content = fp.read()
    cu = Skylake(env="test")
    result = cu.image_binary_upload(content, business_name="ssdspider", scene_name="tupian")
    fp.close()
    print(result)

def test_byte_file():
    """二进制图片上传, test_image_url需要绝对路径"""
    test_image_url = './skylake_test.jpg'
    cu = Skylake(env="test")
    result = cu.file_upload(test_image_url)
    print(result)


