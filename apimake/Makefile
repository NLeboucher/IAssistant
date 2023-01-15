# makefile for the api folder

# sudo apt install gcp

build: #builds the docker image api folder
	cp ../base/pretrainedBiLSTM.py api/ |\
	cp ../base/pretrainedmpnet.py api/ |\
	cp ../base/main.py api/ |\
	docker build .
#	gcp -rv ../base/mpnet-base-snli-mnli/ . |\
#	gcp -rv ../base/snli/ api/ 
#   cp -r ../.data/ api/ 
#	gcp -rv ../base/tut1-model.pt .

run: #runs the docker image from the image built
	@read -p "Enter container image name:" module; \
	docker run -p 8000:8000 -t $$module
clear: #clears the api folder
	# chmod 666 
	yes | rm api/pretrainedBiLSTM.py |\
	yes | rm api/pretrainedmpnet.py |\
	yes | rm api/main.py
#	yes | rm -r mpnet-base-snli-mnli/
#   yes | rm tut1-model.pt
#   yes | rm -r api/.data/
#	yes | rm -r api/snli/
#	gcp -rv ../.vector_cache api/
# yes | rm -r api/.vector_cache/