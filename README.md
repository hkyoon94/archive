### Transformer Inference Monitor

 - Symbolic Music Generation용 Transformer 모델의 auto-regressive generation 과정을 정확히 트래킹하고 모니터링하기 위해 제작한 패키지. 주어진 토큰 시퀀스에 대해 토큰 생성 시의 누적확률분포, 어텐션 맵, 샘플링 엔트로피 및 Groove Similarity 등의 메트릭을 수집할 수 있다.

 - 사용법 및 기능은 패키지 내 `sequence_interpretation_manual.ipynb`파일에 기록되어 있습니다.

---
### Conditional-Diffusion From Scratch

- CIFAR10 데이터셋을 사용한 DDPM 훈련 및 DDPM / DDIM의 batch parallel 인퍼런스를 직접 구현해 본 실험.

---
### Multi-Node Distributed

- 간단한 toy example을 통해, single 및 multi-node에서의 distributed training 방법론들을 기록한 예제.

---
### NRSTDP
- Program archive for article '[An STDP-based encoding method for associative and composite data](https://www.nature.com/articles/s41598-022-08469-6)'.

- `MATLAB` sources:
    - File `image_group.m` is for the simulation of 'retrieval of grouped images' section in the article. Basically performs association of five 32 x 32 pixels of orchestral instrument images used in the article. To raw run this file, please include the following 6 images files 'violin.png', 'trumpet.png', 'harp.png', 'piano.png', 'timpani.png', and 'forest.png', and 'gsprocess.m' in the folder containing this file. User can arbitrarily replace given images and parameters such as the dimension of the images, etc.

    - File `composite_struct.m` is for the simulation of 'multiple groups of memory with composite structure' section in the article. Basically uses the same sentences S1, S2, and S3 in the article. Please include 'gsprocess.m' in the folder containing this file. User can randomly replace the structure of given sentences and perform the same tasks of retrieval.

    - File `gsprocess.m` is a simple function file conducting Gram-Schmidt process for the set of column vectors of an arbitrary matrix.

- `Python` source:
    - `image_retrieval.ipynb` provides a convenient Python demonstrations for MATLAB source `image_group.m`.
