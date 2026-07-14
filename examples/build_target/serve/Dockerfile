# cage artifact-validation serve, containerized.
#
# Runs the always-on boot-check validator. Give it the host docker socket and the
# designated location bind-mounted at the SAME path, so the boot-check can build +
# boot the agent's target on the host daemon and probe it:
#
#   docker run -d --name cage-serve \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     -v /srv/cage_target:/srv/cage_target \        # SAME path host<->container
#     -p 8911:8911 cage-serve \
#     --benchmark /cage/examples/build_target/benchmark.py \
#     --sample my_first_pwnie --root /srv/cage_target

# docker CLI + compose plugin come from the local image — no network needed.
FROM docker:29.1.5-cli AS dockercli

FROM python:3.12-slim
# HTTP(S)_PROXY are build-time only (pip egress); they do not persist to runtime.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=dockercli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose

WORKDIR /cage
COPY pyproject.toml README.md /cage/
COPY cage/ /cage/cage/
RUN pip install --no-cache-dir /cage

# The benchmark package the validator serves (datasets are excluded by
# .dockerignore; the validator takes --internal-port instead, so it doesn't need
# the sample files).
COPY examples/build_target/benchmark.py /cage/examples/build_target/benchmark.py
COPY examples/build_target/prompts/ /cage/examples/build_target/prompts/
COPY examples/build_target/serve/ /cage/examples/build_target/serve/

EXPOSE 8911
ENTRYPOINT ["python3", "/cage/examples/build_target/serve/validator.py"]
