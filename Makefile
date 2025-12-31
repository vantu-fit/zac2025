run:
	MSYS2_ARG_CONV_EXCL="*" docker run --gpus all --rm -it \
		-v "$(PWD)/data/public_test/public_test:/data" \
		-v "$(PWD)/result:/result" \
		--env-file .docker.env \
		zac2025:v1 \
		python3 /code/predict.py


run-bash:
	MSYS2_ARG_CONV_EXCL="*" docker run --gpus all --rm -it \
		-v "$(PWD)/data/public_test/public_test:/data" \
		-v "$(PWD)/result:/result" \
		--env-file .docker.env \
		zac2025:v1 \
		/bin/bash


build:
	docker build -t zac2025:v1 .