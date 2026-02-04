```
 _____ _____ _____ _____ _____ _____ _____ _____ _____     ___ ____  
|   __|   __|   __|   __|     | __  |     |   __| __  |___|_  |    \ 
|__   |   __|  |  |   __|  |  |    -| | | |   __|    -|___|_  |  |  |
|_____|_____|_____|__|  |_____|__|__|_|_|_|_____|__|__|   |___|____/ 
```

# I - Introduction

This repository implements and improves the existing project of Segformer-3D that 
you can found here : https://github.com/OSUPCVLab/SegFormer3D. It is related to 
the research paper on 3D image segmentation cited [here](#Research-Paper).

# II - How to use

First, you will need to clone this repository. It contains almost everything you
need. It only misses the data that you will want to use.
Then, you will need to build the docker using `docker compose build`. This requires
that you already have docker installed. The build can last up to 20min if you don't
already have the image `vllm/vllm-openai:v0.9.1`. And the total size of the docker
image is around 40GB. When done building, if you try to build again, this time it
is only a matter of seconds.
Now, your image is built and you can use for different purposes. The base command
is `docker compose run app`. But as it is, it will not do anything. You have to
pass arguments. Here is the full command `docker compose run app -h -p -t -i -e <number>`.
- `-h` : Using this option will display how the other options work.
- `-p` : Using this option will launch the preprocessing of your data.
- `-t` : Using this option will start the training of a new model on your data.
- `-i` : Using this option will run a given model on your data.
- `-e <number>` : Using this option will evaluate a model. The argument <number>
allows to specify which data is evaluated. If you give `-1`, the model will use
all of your data. If you give a positive integer, the model will use the volume
associated with this number.
Here are some examples of usage : 
-  `docker compose run app -h` : Show the help message.
-  `docker compose run app -p -t` : Launch the preprocessing and then start training 
the model.
-  `docker compose run app -t -e -1` : Train the model and the evaluate it on all 
the data.

# V - Research Paper

```bibtex
@inproceedings{perera2024segformer3d,
  title={SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation},
  author={Perera, Shehan and Navard, Pouyan and Yilmaz, Alper},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4981--4988},
  year={2024}
}
```