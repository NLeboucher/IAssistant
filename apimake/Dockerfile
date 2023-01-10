#We base the image on the official Pytorch image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
#Does not support gpu

#Road to support gpu
# RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

# RUN  apt-get update \
#   && apt-get install -y wget \
#   && rm -rf /var/lib/apt/lists/*

# ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Europe/Paris
# RUN apt-get update && apt-get install -y tzdata

# RUN apt-get update -y \
#   && apt-get install -y \
#     libxml2-dev \
#     php-soap \
#   && apt-get clean -y 

# RUN apt -yqq update 
# RUN apt-get install -y php php-fpm php-pdo-mysql php-mbstring php-mysqli

#install cuda
#RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
#RUN sh cuda_11.6.0_510.39.01_linux.run

#add the api running scripts
ADD api/main.py .
ADD api/pretrainedBiLSTM.py .
ADD api/pretrainedmpnet.py .
ADD api/.data/ .

#ADD api/.vector_cache/ .
#add the pretrained models
ADD tut1-model.pt .
ADD mpnet-base-snli-mnli-finetuned-mnli .

#add requirements
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

#add the api
CMD uvicorn main:app --host 0.0.0.0 --port 8000