FROM continuumio/miniconda3

RUN apt-get update && apt-get install gcc --yes && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/* 


WORKDIR /app/deepfake
COPY . .

RUN conda env create -n deepfake -f /app/deepfake/environment_nano.yml

SHELL ["conda", "run", "-n", "deepfake", "/bin/bash", "-c"] 

RUN echo "source activate deepfake" > ~/.bashrc
ENV PATH /opt/conda/envs/deepfake/bin:$PATH

EXPOSE 8000

CMD [ "python", "./manage.py", "migrate"]
CMD [ "python", "./manage.py", "runserver", "0.0.0.0:8000"]
