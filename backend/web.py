
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, make_response, request
from se_sample import search_api
import json
app = Flask(__name__)



@app.route('/search/<query>')
def search(query):
    result = search_api(query)
    re = {"docs":result}
    rst = make_response(json.dumps(re))
    rst.headers['Access-Control-Allow-Origin'] = '*'
    return rst



if __name__ == '__main__':
    app.run(host="localhost", port=4000,debug=True)
