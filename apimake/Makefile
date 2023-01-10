# makefile for the api folder

# sudo apt install gcp

build: #builds the docker image api folder
	cp ../my2/pretrainedBiLSTM.py api/ |\
	cp ../my2/pretrainedmpnet.py api/ |\
	# gcp -rv ../my2/tut1-model.pt . |\
	gcp -rv ../my2/mpnet-base-snli-mnli-finetuned-mnli/ api/ |\
	cp ../my2/main.py api/ |\
	gcp -rv ../my2/snli/ api/ |\
	cp -r ../.data/ api/ |\
	docker build .
dockerrun: #runs the docker image from the image built
	@read -p "Enter Module Name:" module; \
	docker run -p 8000:8000 -t $$module
clear: #clears the api folder
	yes | rm api/pretrainedBiLSTM.py |\
	yes | rm api/pretrainedmpnet.py |\
	# yes | rm tut1-model.pt |\
	yes | rm -r api/mpnet-base-snli-mnli-finetuned-mnli/ |\
	yes | rm api/main.py |\
	yes | rm -r api/.data/ |\
#	yes | rm -r api/snli/
#	gcp -rv ../.vector_cache api/ |\
# yes | rm -r api/.vector_cache/ |\