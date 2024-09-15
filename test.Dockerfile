# Convenience file to emulate a server with no GPU.

# sudo docker build -f test.Dockerfile -t test .
# sudo docker run -v $(pwd):/workspace -e SCUDA_SERVER=172.17.0.1 -e LD_PRELOAD=/workspace/libscuda.so test nvidia-smi --query-gpu=name --format=csv
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y nvidia-utils-550

CMD ["nvidia-smi"]