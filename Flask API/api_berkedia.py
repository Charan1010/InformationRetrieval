from flask import Flask,request,jsonify
from http import HTTPStatus as status
from guide_berkedia import wrapper as wrapper1
from ticket_IR import wrapper as wrapper2
import numpy as np
import pathlib
import urllib
import cProfile, pstats,io
import pytest
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


app = Flask(__name__)


@app.route('/IR/guide',methods=['GET','POST'])
def get_guide():
    try:
        req = request.json
        query=req['query']
    except Exception as e:
        return "Bad Request:Input Empty",status.BAD_REQUEST
    try:
        if not isinstance(query, str):
            raise TypeError('Please provide a string argument')
    except TypeError:
        return "Wrong Format,please provide a string",status.UNSUPPORTED_MEDIA_TYPE


    pr = cProfile.Profile()
    pr.enable()
    (results,page_num,confidences)=wrapper1(query,1)
    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    #print(s.getvalue())
    print("AT THE END,RETURNING JSON")
    return jsonify({'results':results,'confidences':confidences})

@app.route('/IR/ticket',methods=['GET','POST'])
def get_ticket():
    try:
        req=request.json
        query=req['query']
    except Exception as e:
        return "Bad Request:Input Empty",status.BAD_REQUEST
    try:
        if not isinstance(query, str):
            raise TypeError('Please provide a string argument')
    except TypeError:
        return "Wrong Input Format,please provide a string ",status.UNSUPPORTED_MEDIA_TYPE

    pr = cProfile.Profile()
    pr.enable()
    (confidences,resolutions)=wrapper2(query)
    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print("AT THE END,RETURNING JSON")
    #print(s.getvalue())
    return jsonify({'confidences':confidences,'resolutions':resolutions})
@app.route('/IR/tests')
def run_tests():
    pytest.main(['-p','no:warnings'])
    return 'OK'

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5005)

