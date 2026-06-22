# Reproducible image for spatial-hotspots.
# Uses a micromamba base so the geospatial stack (GDAL/GEOS/PROJ) installs
# cleanly from conda-forge instead of fighting pip wheels.
FROM mambaorg/micromamba:1.5.8

WORKDIR /app

# Install dependencies first for better layer caching.
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /app/requirements.txt
RUN micromamba install -y -n base -c conda-forge \
        python=3.12 \
        numpy pandas pyyaml requests \
        geopandas shapely libpysal esda splot mapclassify matplotlib-base \
        mgwr pytest ruff mypy \
    && micromamba clean --all --yes

# Copy the project and install it into the base env.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-deps -e .

# Default: run the test suite (the pure-numpy Moran's I known-answer tests).
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["pytest", "-q"]
