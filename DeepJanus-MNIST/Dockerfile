# this is our first build stage, it will not persist in the final image
FROM ubuntu as download-deepjanus-mnist-private

# install git
RUN apt-get update
RUN apt-get install -y git

# add credentials on build
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/

# Add the keys and set permissions
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa && \
    chmod 700 /root/.ssh/id_rsa

# make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Download the lastest from the repo and show the version on console
RUN git clone git@github.com:zohdit/DeepHyperion.git deepjanus_mnist && \
    cd deepjanus_mnist && git log --pretty=oneline

# 
FROM python:3.6

# copy the repository form the previous image
# TODO Maybe we need to get rid of .git ?
COPY --from=download-deepjanus-mnist-private /deepjanus_mnist /deepjanus_mnist


# install all the system deps
RUN apt-get update
RUN apt-get install -y git build-essential python-dev libagg-dev libpotrace-dev pkg-config
RUN apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
RUN apt-get install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 librsvg2-dev

# Prepare pypotrace
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install numpy

RUN git clone https://github.com/flupke/pypotrace.git
RUN cd pypotrace && \
    pip install .

RUN cd /deepjanus_mnist && \
    pip install -r requirements.txt

# For now the MODEL and the PROPERTIES are hardcoded in the image under /deepjanus_mnist as the 
WORKDIR /deepjanus_mnist
CMD ["/usr/local/bin/python", "main.py"]