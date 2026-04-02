# Cold Diffusion from scratch
Implementing from scratch ["Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise"](https://arxiv.org/abs/2208.09392) NeurIPS 2023 

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/cold_diffusion_from_scratch.git
pip install -r requirements.txt
```

### Train
``` 
python train.py cold_diffusion/config/fashion_mnist_config.yaml 
```

### Qualitative Results
> Qualtitative results of the model trained on [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) for ~20 epochs. Initial seed-noise distribution derived from random validation samples blurred for 300 iterations. 

https://github.com/user-attachments/assets/eaf316be-6bae-4c9d-be1c-c4f6e6670ff7
