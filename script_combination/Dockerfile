FROM ubuntu:latest


RUN apt-get update
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN apt-get install -y nano


RUN pip3 install -U dm_control
RUN pip3 install -U gym==0.26.2
RUN pip3 install -U pygame==2.1.0
RUN pip3 install -U imageio
RUN pip3 install -U pandas
RUN pip3 install -U PyYaML
RUN pip3 install -U seaborn
RUN pip3 install -U scikit-image

RUN apt-get install -y libglfw3
RUN apt-get install -y libglew2.2
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libosmesa6


RUN pip3 install -U opencv-python
RUN apt-get install -y libglib2.0-0


RUN pip3 install -U torch
RUN pip3 install -U torchaudio
RUN pip3 install -U torchvision


WORKDIR /
RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
RUN cd cares_reinforcement_learning && \
    python3 -m pip install --editable .
WORKDIR /

