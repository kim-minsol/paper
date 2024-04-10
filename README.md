# paper

논문을 읽고, 관련 모델을 직접 따라 구현해보는 레포입니다.   

Genarative 모델이나 NeRFs를 주로 다룰 예정입니다.


# Geneartive

* ### VAE
**Authors**: Diederik P. Kingma, Max Welling   
**Year**: 2013   
**Paper Link**: https://arxiv.org/pdf/1312.6114.pdf   
**Descriptions**: Autoencoder 구조를 이용하여 generative task에 이용된 첫 모델입니다. input data를 latent space로 매핑한 후, decoder를 통해 새로운 데이터를 생성해냅니다. 목적 함수로는 ELBO를 활용하였으며 Bayesian 관점으로 이를 해석하는 내용을 블로그에 남겨두었습니다! 

https://velog.io/@rlaalsthf02/3-Variational-Inference-1


## Diffusions

* ### DDPM (Denoising Diffusion Probabilistic Models) (작성 중)
**Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel 
**Year**: 2020    
**Paper Link**: [https://arxiv.org/pdf/1312.6114.pdf   ](https://arxiv.org/pdf/2006.11239.pdf)   
**Descriptions**: Forward process로 이미지를 std 분포의 noise로 매핑한 후에, Reverse process에서 density model을 활용하여 noise를 denoising하여 이미지를 생성해내는 모델입니다.   

https://velog.io/@rlaalsthf02/Denoising-Diffusion-Probabilitic-ModelsDDPM


**참고 코드**   
* https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main   
* https://github.com/CompVis/latent-diffusion/tree/main

<br>
<br/>

# NeRFs

* ### NeRF (Vanila)
**Authors**: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng   
**Year**: 2020   
**Paper Link**: https://arxiv.org/pdf/2003.08934.pdf   
**Descriptions**: position(xyz) 값과 viewing direction으로 pixel color(rgb)를 예측하는 Fully-connected Network를 사용하여 2D image를 3D 렌더링하는 모델입니다.

https://velog.io/@rlaalsthf02/NeRF


**참고 코드**   
* https://github.com/yenchenlin/nerf-pytorch

<br>
<br/>

# etc Models

* ### ViT (“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”)
**Authors**: Google Research  
**Year**: 2021   
**Paper Link**: https://arxiv.org/pdf/2010.11929.pdf  
**Descriptions**: NLP에서만 활용되던 Transformer 구조를 Vision 분야에 적용시킨 최초의 모델입니다. 이미지를 patch로 쪼개어 임베딩한 후 (+ position embedding) self-attention을 통해 학습합니다.

https://velog.io/@rlaalsthf02/ViT


<br>
<br/>

## plan
* NeRF Done!
* vae 코드 수정
* ddpm 코드 구현 (작성 중) 현재 Gaussian diffusion 까지는 구현 완료!
* DDIM
* instant NGP
* LDM