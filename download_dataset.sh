#!/bin/bash

if [[ ! -d $1 ]]; then
	mkdir $1
fi

cd $1

if [[ ! -d original_high_fps_videos ]]; then
	if [[ ! -f DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip ]]; then
		wget http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
	fi

	unzip DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
	rm DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
fi

cd original_high_fps_videos

for filename in *.mov *.mp4 *.MP4 *.m4v *.MOV ; do
	if [[ ! -d ${filename%.*} ]]; then
		mkdir ${filename%.*}
	fi

	if [[ -z "$(ls -A ${filename%.*})" ]]; then
		ffmpeg -i ${filename} "${filename%.*}"/%05d.png
		rm ${filename}
	fi
done

cd ..

if [[ ! -d NFS ]]; then
	mkdir NFS
fi

cd NFS

if [[ -z "$(ls -A .)" ]]; then
	curl -fSsl http://ci2cv.net/nfs/Get_NFS.sh | bash -
fi

for filename in *.zip ; do
	if [[ ! -d ${filename%.*} ]]; then
		unzip -j ${filename} ${filename%.*}/240/${filename%.*}/* -d ./${filename%.*}
		rm ${filename}
	fi
done