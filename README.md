# Chinese Character-Level Language Model
http://arxiv.org/abs/1410.4615

### Requirements
tensorflow 0.7.1

### Docker
```
# Remove docker containers
$ docker rm $(docker ps -a -q)

# Remove docker images
$ docker rmi $(docker images -f "dangling=true" -q)

# Build image
$ docker build -t char-rnn .

# Run container in detach mode
$ docker run --name char-rnn --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it -d --privileged -v /home/core/chinese-char-rnn/data:/src/data -v /home/core/chinese-char-rnn/checkpoint:/src/checkpoint -v /home/core/chinese-char-rnn/log:/src/log char-rnn /bin/bash -c "source /venv/bin/activate && /venv/bin/python /src/train.py"

# Attach container
$ docker attach char-rnn

# Run container in exec mode
$ docker run --name char-rnn --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm --privileged -v /home/core/chinese-char-rnn/checkpoint:/src/checkpoint char-rnn bash

# Detach
[Ctrl + p] + [Ctrl + q]
```
