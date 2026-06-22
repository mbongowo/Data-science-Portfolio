# Reproducible runtime for eo-monitor using pixi on a slim Debian base.
FROM ghcr.io/prefix-dev/pixi:0.27.1 AS build

WORKDIR /app

# Resolve and install the locked environment first for better layer caching.
COPY pixi.toml pyproject.toml README.md ./
COPY src/ ./src/
RUN pixi install --locked || pixi install

# Bake an activation script so the entrypoint runs inside the pixi env.
RUN pixi shell-hook -e default > /shell-hook.sh && \
    echo 'exec "$@"' >> /shell-hook.sh

FROM ghcr.io/prefix-dev/pixi:0.27.1 AS runtime
WORKDIR /app
COPY --from=build /app /app
COPY --from=build /shell-hook.sh /shell-hook.sh
COPY config/ ./config/

ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]
CMD ["eo-monitor", "run", "--config", "config/corn_belt.yaml"]
