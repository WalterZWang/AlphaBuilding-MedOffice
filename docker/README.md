This directory contains files for building docker images used for co-simulation of
the ``AlphaBuilding`` project.  The docker images are split into four as follows:

``der_base`` builds from Docker's ``ubuntu:16.04``, installs basic software and
python packages, and creates a new user: developer.

``der_jmodelica`` builds from ``der_base`` and installs JModelica for co-simulation. (reused from the DER project), pushed to ``docker.hub`` as ``walterzwang/jmodelica:latest``

``eplus:9.2`` builds from ``der_jmodelica`` and installs EnergyPlus 9.2, pushed to ``docker.hub`` as ``walterzwang/eplus:9.2``

``drl_eplus:9.2`` builds from ``eplus:9.2`` and installs gym, PyTorch, TensorBoardX and etc., pushed to ``docker.hub`` as ``walterzwang/drl_eplus:9.2``

To build the images:

1. ``$ cd base`` then ``$ make build``.  Takes around 8 mins.
2. ``$ cd jmodelica`` then ``$ make build``.  Takes around 60 mins.
3. ``$ cd eplus`` then ``$ make build``.  Takes around 3 mins.
4. ``$ cd drl`` then ``$ make build``.  Takes around 3 mins.

To run the test script:

1. ``$ cd drl``, revise parameter VOLUMN_PATH in the makefile, then ``$ make run``
2. ``$ cd virtualEnv/``
3. ``$ python test.py`` to run the test script
