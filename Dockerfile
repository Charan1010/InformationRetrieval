# Dockerfile - this is a comment. Delete me if you want.

FROM python:3.6

COPY . /IR_dock

WORKDIR /IR_dock

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["api_berkedia.py"]
