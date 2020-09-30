FROM python:3
WORKDIR /project
ADD . /project


RUN pip install -r requirements.txt


EXPOSE 5000
CMD ["python","load_test_dataset.py"]
CMD ["python","app.py"]
