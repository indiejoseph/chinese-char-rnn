FROM tflowgpu
MAINTAINER Joseph Cheng <indiejoseph@gmail.com>

RUN mkdir /src

WORKDIR /src

ADD requirements.txt /src

RUN pip install virtualenv

# Create Python environment
RUN /usr/local/bin/virtualenv --system-site-packages /venv --distribute
RUN /venv/bin/pip install --upgrade pip
RUN /bin/bash -c "source /venv/bin/activate;"
RUN /venv/bin/pip install -r requirements.txt

ADD . /src

# CMD /bin/bash -c "source /venv/bin/activate && /venv/bin/python /src/train.py"
