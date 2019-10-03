FROM  pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

COPY . /workspace/VLP

# set python3 to default
RUN echo "alias python=python3" >> /root/.bashrc && \
    bash /root/.bashrc

# install dependencies
RUN apt update -y
RUN apt install wget vim zip unzip ca-certificates-java openjdk-8-jdk -y

WORKDIR /workspace/VLP

RUN bash ./setup.sh

# install python libraries
RUN pip install tensorboardX six numpy tqdm pandas scikit-learn py-rouge matplotlib scikit-image h5py demjson torchtext stanfordnlp # --user
RUN pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval # --user

RUN cd ./coco-caption && ./get_stanford_models.sh

# detectron fc7 weights
RUN wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz && tar -zxvf detectron_weights.tar.gz && rm detectron_weights.tar.gz

RUN cd ./pythia && \
    mkdir -p data && cd data && \
    wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz && \
    tar xf vocab.tar.gz && rm vocab.tar.gz

RUN cd ./pythia/data && wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && \
    unzip v2_Annotations_Val_mscoco.zip && rm v2_Annotations_Val_mscoco.zip

RUN cd ./pythia/data && mkdir -p imdb && cd imdb && \
    wget https://dl.fbaipublicfiles.com/pythia/data/imdb/vqa.tar.gz && \
    tar xf vqa.tar.gz && rm vqa.tar.gz

EXPOSE 8888
