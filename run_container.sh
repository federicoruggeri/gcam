docker run --runtime=nvidia --gpus all -d --mount type=bind,src=/"$(pwd)/gcap",target=/gcap -it --name $1 gcap
