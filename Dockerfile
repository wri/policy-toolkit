FROM jupyter/scipy-notebook

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python-dev python-setuptools


RUN pip install --upgrade pip && pip install \
    pytesseract \
    Wand \
    pandas \
    snorkel-metal \
    tensorboardX\
    git+https://github.com/HazyResearch/snorkel \
    sqlalchemy \
    matplotlib \
    spacy \
    lxml \
    treedlib \
    numba \
    nltk \
    numbskull \
    regex

# copy everything from the component folder into /workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["python", "src/main.py"]
