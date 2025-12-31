run-docker:
	docker run --gpus all --rm -it \
		-v $(PWD)/data/public_test/public_test:/data \
		-v $(PWD)/result:/result \
		--env-file .docker.env \
		pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel \
		/bin/bash

run: 
	docker run --gpus all --rm -it \
		-v ./data_infer:/data \
		-v ./result:/result \
		--env-file .docker.env \
		zac2025:v1 \
  		python3 /code/predict.py