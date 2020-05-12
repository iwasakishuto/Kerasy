help:
	@cat Makefile

DATA?="${HOME}/Data"
DOCKER_FILE=Dockerfile
DOCKER=docker
TEST=tests/
HOST_PORT=8888
SRC?=$(shell dirname `pwd`)
EXAMPLES_PATH=$(SRC)/examples

build:
	docker build -t kerasy -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data kerasy bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data kerasy ipython

notebook: build
	$(DOCKER) run -it -p $(HOST_PORT):8888 -v $(SRC):/src/workspace -v $(DATA):/data kerasy

examples: build
	$(DOCKER) run -it -p $(HOST_PORT):8888 -v $(SRC):/src/workspace -v $(EXAMPLES_PATH):/data/kerasy_examples kerasy

rebuild:
	docker build -t kerasy --no-cache -f $(DOCKER_FILE) . 
