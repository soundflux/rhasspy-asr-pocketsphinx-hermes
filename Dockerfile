# ARG BUILD_ARCH=amd64
# FROM ${BUILD_ARCH}/debian:buster-slim
# ARG BUILD_ARCH=amd64

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install --no-install-recommends --yes \
#     libgfortran3 sox

# # Install pre-built mitlm
# ADD mitlm-0.4.2-${BUILD_ARCH}.tar.gz /
# RUN mv /mitlm/bin/* /usr/bin/
# RUN mv /mitlm/lib/* /usr/lib/

# # Install pre-built phonetisaurus
# ADD phonetisaurus-2019-${BUILD_ARCH}.tar.gz /usr/

# COPY pyinstaller/dist/* /usr/lib/rhasspyasr_pocketsphinx_hermes/
# COPY debian/bin/* /usr/bin/

# ENTRYPOINT ["/usr/bin/rhasspy-asr-pocketsphinx-hermes"]

ARG BUILD_ARCH=amd64
FROM ${BUILD_ARCH}/python:3.7-alpine as build

RUN apk add --no-cache build-base swig

ENV VENV=/usr/.venv

RUN python3 -m venv $VENV
RUN $VENV/bin/pip3 install --upgrade pip

COPY requirements_rhasspy.txt requirements.txt /tmp/
RUN $VENV/bin/pip3 install https://github.com/synesthesiam/pocketsphinx-python/releases/download/v1.0/pocketsphinx-python.tar.gz
RUN $VENV/bin/pip3 install -r /tmp/requirements_rhasspy.txt
RUN $VENV/bin/pip3 install -r /tmp/requirements.txt

# -----------------------------------------------------------------------------

ARG BUILD_ARCH=amd64
FROM ${BUILD_ARCH}/python:3.7-alpine

RUN apk add --no-cache sox

WORKDIR /usr
COPY --from=build /usr/.venv /usr/.venv/

COPY **/*.py rhasspyasr_pocketsphinx_hermes/

ENTRYPOINT ["/usr/.venv/bin/python3", "-m", "rhasspyasr_pocketsphinx_hermes"]