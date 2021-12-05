ARG OWNER=jupyter
FROM $OWNER/tensorflow-notebook
WORKDIR /hamidadesokan/notebook
USER root
# COPY . .
RUN set -ex \
      && conda install -c pytorch --quiet --yes \
      'pytorch' \
      'torchvision' \
     && conda install -y -c plotly --quiet --yes plotly \
     && conda install -y 'pip'  \
     && conda install -y 'ipykernel'\
     && conda install -y jupytext -c conda-forge


