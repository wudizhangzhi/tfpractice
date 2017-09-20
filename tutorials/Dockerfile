From frolvlad/alpine-python3

WORKDIR /home/zzc/tfpractice/tutorials/imagenet/

ADD . /home/zzc/tfpractice/tutorials/imagenet/

RUN pip install -r requirements.txt

EXPOSE 6006

CMD ["python", "Cifar_10.py", "--maxstep=1000000", "--decay_per_step=50000", "is_save=True"]
