run-docker:
	docker run --gpus all --rm -it \
		-v $(PWD)/data:/data \
		-v $(PWD)/result:/result \
		--env-file .docker.env \
		pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel \
		/bin/bash