#!/bin/env bash

export DOCKER_BUILD_ARGS="--build-arg ftp_proxy=${ftp_proxy} --build-arg FTP_PROXY=${FTP_PROXY} --build-arg http_proxy=${http_proxy} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg https_proxy=${https_proxy} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg no_proxy=${no_proxy} --build-arg NO_PROXY=${NO_PROXY} --build-arg socks_proxy=${socks_proxy} --build-arg SOCKS_PROXY=${SOCKS_PROXY}"
#export DOCKER_BUILD_ARGS="--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}"

export DOCKER_RUN_ENVS="--env ftp_proxy=${ftp_proxy} --env FTP_PROXY=${FTP_PROXY} --env http_proxy=${http_proxy} --env HTTP_PROXY=${HTTP_PROXY} --env https_proxy=${https_proxy} --env HTTPS_PROXY=${HTTPS_PROXY} --env no_proxy=${no_proxy} --env NO_PROXY=${NO_PROXY} --env socks_proxy=${socks_proxy} --env SOCKS_PROXY=${SOCKS_PROXY}"

#export PYTORCH_VERSION=v1.90

VERSION=4.0
export IMAGE_NAME=mlperf_inference_3dunet:${VERSION}


echo "Building 3d-unet-99.9 workflow container"

DOCKER_BUILDKIT=1 docker build ${DOCKER_BUILD_ARGS} -f Dockerfile -t ${IMAGE_NAME} ../../../..

# docker push ${IMAGE_NAME}
