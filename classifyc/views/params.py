import datetime as dt
import re
from urllib import parse

from flask import request
from marshmallow import Schema, fields, post_load, validates, ValidationError, pprint
from werkzeug.exceptions import HTTPException


class BytesField(fields.Field):
    def _validate(self, value):
        if not isinstance(value, bytes):
            raise ValidationError('Invalid input type.')

        if value is None or value == b'':
            raise ValidationError('Invalid value')


def unmarshal_json(schema):
    from flask import request
    from views.exceptions import ParamsValidationError
    try:
        if request.method == 'POST':
            js = request.get_json()
        else:
            image_url = request.args.get("pic")
            if image_url is None:
                raise ParamsValidationError(400, "no args pic url", 400)
            js = {'url': image_url}

        instance = schema().load(js)
        return instance
    except ValidationError as err:
        raise ParamsValidationError(400, str(err.messages), 400)
    except HTTPException as e:
        raise ParamsValidationError(400, str(e), 400)

class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.created_at = dt.datetime.now()

    def __repr__(self):
        return f'<User(name={self.name}>'


class UserSchema(Schema):
    name = fields.Str()
    email = fields.Email()
    create_at = fields.DateTime()

    @post_load
    def make_user(self, data, **kwargs):
        return User(**data)


class Picture:
    def __init__(self, image_bytes=None, url=None):
        self.url = url
        self.image_bytes = image_bytes

    def __repr__(self):
        return f'<Picture(url={self.url}>'


class PictureSchema(Schema):
    url = fields.URL()
    image_bytes = BytesField(allow_none=True, required=False)

    @validates('url')
    def validate_url(self, url):
        import requests
        try:
            image_url = parse.unquote(url)
            if (image_url.find('cdn.weipaitang.com') >= 0 or image_url.find('cdn.wptqc.com') >= 0) and (
                    not re.search('/w/\d+', image_url.lower())):
                try:
                    width = int(re.search('W(\d+)H(\d+)', image_url.upper()).group(1))
                    if width > 1080:
                        image_url = image_url + '/w/640'
                except AttributeError:
                    pass
            self.image_bytes = requests.get(image_url, timeout=10).content
        except requests.exceptions.RequestException:
            raise ValidationError('Image url is invalid, please check.')
        else:
            if not self.image_bytes:
                raise ValidationError('Image url is invalid, please check.')

    @post_load
    def make_picture(self, data, **kwargs):
        return Picture(self.image_bytes, **data)
