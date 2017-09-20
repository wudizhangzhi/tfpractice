From tensorflow/tensorflow 

WORKDIR /app 

ADD . /app

#RUN pip install -r requirements.txt

EXPOSE 6006

#CMD ["python", "Cifar_10.py", "--maxstep=1000000", "--decay_per_step=50000", "is_save=True"]
ENTRYPOINT ["python", "Cifar_10.py"]
