# This dockerfile creates user, and installs python

FROM walterzwang/eplus:9.2

# Install required python packages with root

USER root
# ENV HOME /home/developer


# Install base python packages
RUN python -m pip install --upgrade pip && \
    python -m pip install --user \
    python-dateutil==2.6.1 \
	pytz==2017.2 \
	scikit-learn==0.18.2 \
	sphinx==1.6.3 \
	numpydoc==0.7.0 \
	tzwhere==2.3 \
	influxdb==5.2.2 \
	pyyaml==3.13 \
	gym \
	requests==2.21.0 \
	pydoe  \
	torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html  \
	tensorboardX

# WORKDIR ${HOME}


