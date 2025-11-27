# use the miniforge base, make sure you specify a verion
FROM condaforge/miniforge3:latest

# copy the lockfile into the container
COPY conda-lock.yml conda-lock.yml

# setup conda-lock and install packages from lockfile
RUN conda install -n base -c conda-forge conda-lock -y
RUN conda-lock install -n group30-522 conda-lock.yml

# expose JupyterLab port
EXPOSE 8888

# sets the default working directory
# this should match the path where the repo is mounted in docker-compose
WORKDIR /workspace

# run JupyterLab on container start
# uses the jupyterlab from the install environment
CMD ["conda", "run", "--no-capture-output", "-n", "group30-522", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]