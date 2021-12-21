ARG PYTHON_VERSION=3.9

FROM python:${PYTHON_VERSION}-slim as builder

WORKDIR /home/app

RUN python -m venv /opt/venv

COPY ./requirements.txt ./

ENV PATH=/opt/venv/bin:${PATH}

RUN pip install wheel

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip pip install -r ./requirements.txt

COPY ./logger ./logger

RUN cd ./logger && \
    pip install .

COPY ./core ./core

COPY ./setup.py ./

RUN pip install .

FROM python:${PYTHON_VERSION}-slim

ENV TZ="Europe/Moscow"

# RUN apt update && \
#     apt install -y --no-install-recommends libpq5 && \
#     apt remove

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:${PATH}

WORKDIR /home/app

COPY ./config.yaml ./config.test.yaml ./log_settings.yaml ./test_functionality.py ./download_model.py ./

RUN python ./download_model.py

COPY ./bot.py ./

ENV LOG_CONFIG=./log_settings.yaml

CMD python ./bot.py 