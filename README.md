# DRAEM : A discriminatively trained reconstruction embedding for surface anomaly detection

Topic: 결함 탐지, 컴퓨터 비전
Year: 2021

[](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf)

# 0. Abstract

***Recent surface anomaly detection methods***

⇒ *generative models* 

1. *to accurately reconstruct the normal areas*
2. to *fail on anomalies*

*we cast surface anomaly detection primarily as a discriminative problem*

and *propose a `**discriminatively trained reconstruction anomaly embedding model**` (DRÆM)*

***DRÆM learn the contents below***

1. *a joint representation of an anomalous image*
2. *its anomaly-free reconstruction*
3. *a decision boundary between normal and anomalous examples*

***The method enables the contents below***

1. *direct anomaly localization without the need for additional complicated post-processing of the network output*
2. *can be trained using simple and general anomaly simulations*

[https://github.com/VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM)

# 1. Introduce

**general anomaly detection problem**

⇒ considers anomalies as *`entire images`* that significantly differ from the non-anomalous training set images

**surface anomaly detection problems(ours)**

⇒ the anomalies occupy only *`a small fraction* of image` pixels and are typically close to the training set distribution

### Reconstructive Methods

Reconstructive methods, such as Autoencoders [5, 1, 2, 26] and GANs [24, 23], have been extensively explored since they enable learning of `a powerful reconstruction` subspace, using `only anomaly-free images.`

Relying on poor re-construction capability of anomalous regions, not observed in training,

the anomalies can then be detected `by thresholding the difference between the input image` and `its reconstruction.`

**Issue**

determining the presence of anomalies that are not substantially different from normal appear- ance remains challenging, since these are `often well reconstructed`.

**Recent improvements**

1. **The difference**
by thresholding the difference between `the input image` and `its reconstruction.`
    
    ⇒ the difference between `deep features extracted from a general-purpose network` and `a network specialized for anomaly-free images`
    
2. **Discrimination**
: Discrimination can also be formulated as a `deviation from a dense clustering of non-anomalous textures` within the deep subspace
    
    <aside>

    **💡 the deep subspace** : as forming such a compact subspace `prevents anomalies from being mapped close to anomaly-free samples.`
    </aside>
    

### Hypothesis

We hypothesize that over-fitting can be substantially reduced `by training a discriminative model over the joint, reconstructed and original, appearance` along with the `reconstruction subspace`.

간단히 말해, 모델은 원래 데이터와 그 데이터의 재구성 버전 사이의 '거리'를 학습하여, 이를 통해 실제 세계에서 다양한 이상 현상을 더 잘 이해하고 분류할 수 있게 됩니다. 이렇게 하면 모델이 합성 데이터에 과적합되는 문제를 피할 수 있습니다.

# 2. Related work

> Instead of the commonly used image space reconstruction, the reconstruction of pretrained network features `can also be used for surface anomaly detection`
> 
> 
> [Unsupervised anomaly segmentation via deep feature reconstruction](https://www.sciencedirect.com/science/article/abs/pii/S0925231220317951)
> 
> [](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.pdf)
> 
> Anomalies are detected based on the assumption that features of a pre-trained network will not be faithfully reconstructued by another network trained only on anomaly-free images.
> 
> 비정상들은 사전학습된 network의 피쳐들이 anomaly-free 이미지들로만 학습된 다른 network로부터 정확하게 재구성되지 않을 것이라는 가정에 기반하여 탐지된다. 
> 

> Recently `Patch-based one-class classification methods` have been considered for surface anomaly detection.
> 

…

# 3. DREAM
![Untitled](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/54277cb9-8f0e-47a4-a284-2bafb8b7351a)
<br>
The anomaly detection process of the proposed method

## 1) **The reconstructive Sub network**
![Untitled 1](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/9683bc64-e4ce-432a-b5f6-8029f27fce1b)
<br>
The reconstructive sub-network is trained to implicitly detect and reconstruct the anomalies with semantically plausible anomaly-free content, while keeping the non-anomalous regions of the input image unchanged.

| $I$ | original image |
| --- | --- |
| $I_a$ | an artificially corrupted version |

### **Loss**

1. **SSIM(Structural Similrity Index) loss**
    - 주어진 2개의 이미지의 `similarity(유사도)`를 계산하는 측도로 사용
    - `SSIM`은 두 이미지의 단순 유사도를 측정하는데 사용
    - 두 이미지가 유사해지도록 만들어야 되는 문제일 때 `SSIM`을 Loss Function 형태로 사용하기도 합니다. 왜냐하면 `SSIM`이 gradient-based로 구현되어 있기 때문입니다.
    ![Untitled 2](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/b65fc501-610e-441f-8a4c-11045f699d74)
    <br>

    | $H, W$ | 이미지 $I$의 높이 넓이 |
    | --- | --- |
    | $N_p$ | 이미지 $I$의 pixel 수 |
    | $I_r$ | reconstructed  $I$(output) |
    | $SSIM(I, I_r)_{(i, j)}$ | SSIM value for patch of $I$ and $I_r$ |
2. **The reconstruction loss**
    ![Untitled 3](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/3cf67455-443e-496c-8248-73edd1e59fbe)
    <br>
    
    | $\lambda$ | loss balancing hyper-parameter |
    | --- | --- |

### **Summary**

1. **Train**
    
    
    | Input / Target | noise 처리된 정상   /   raw 정상 |
    | --- | --- |
    | Output | reconstructed 된 정상 |
    | What is trained | noise(가상결함)을 정상으로 복원하는 것 |
    | Evaluate cost<br>(L2 + SSIM loss) | reconstruction loss(Target과 Output으로 계산됨)<br>👉 최소화 하는 것이 목표 |
2. **Classification** 
    
    
    | input | raw 비정상 |
    | --- | --- |
    | output | 정상으로 reconstructed 된 비정상 |
    | Classification | anomalies를 정상부위로 바꿈으로서<br>👉 reconstruction loss 높아짐<br>👉 결함 판별 |
    

## 2) **The discriminative sub-network**
![Untitled 4](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/a857b46a-2a86-4feb-8c3a-4cbe45ed352d)

the discriminative sub-network learns a joint reconstruction-anomaly embedding and produces accurate anomaly segmentation maps from the concatenated reconstructed and original appearance.

정상으로 복원하는 The reconstruction network의 특징 때문에

- Anomalous images의 $I_r$은 $I$와 크게 차이 난다.
- 그리고 anomaly segmentation에 필수적인 정보를 제공함

| Input  | $I_c$ : $I_r$ 과 $I$의 채널별 concatenation |
| --- | --- |
| Output | $M_o$ : a pixel-level anomaly detection mask |

### Loss : focal loss

Focal Loss($L_{seg}$) is applied on the discriminative sub-network output `to increase robustness towards accurate segmentation` of hard examples.

`Focal Loss`는 Easy Example의 weight를 줄이고 `Hard Negative Example`에 대한 학습에 초점을 맞추는 Cross Entropy Loss 함수의 확장판이다.

👉  data imbalanceing
![Untitled 5](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/21b032b7-a705-43f6-a1a9-441ac2ee3189)

| $\alpha$ | 전체적인 Loss 값을 조절하는 값 |
| --- | --- |
| $(1 - p_t)^\gamma$ | $\gamma \geq 0$ 의 값을 조절해야 좋은 성능 얻을 수 있음 |
| $\gamma$ | focusing parameter,<br>Easy Example에 대한 Loss의 비중을 낮추는 역할 |
<br>

![Untitled 6](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/8491e80f-7c5a-46b4-83b1-eacecd3af74a)

<br>

$\lambda = 0$ : Cross entropy loss와 같음

### Total Loss
![Untitled 7](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/38e6dfa4-309f-4243-90af-6bc181562554)
<br>

| $M_a$ | ground truth |
| --- | --- |
| $M$ | output segmentation masks |

## 3) Simulated anomaly generation

### A noise image
![Untitled 8](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/c985c7a8-da2e-4096-88ca-be21e5d9916b)

Figure 4. Simulated anomaly generation process. The binary anomaly mask Ma is generated from Perlin noise P . The anomalous regions are sampled from A according to Ma and placed on the anomaly free image I to generate the anomalous image $I_a$.

DRÆM은 대상 영역의 실제 이상 현상을 현실적으로 반영하기 위해 시뮬레이션을 요구하지 않고, 막 분포가 끝난 모양을 생성하여 정상으로부터의 편차를 통해 이상 현상을 인식할 수 있는 적절한 거리 함수를 학습할 수 있도록 합니다.

- `시뮬레이션을 요구안함 적절한 거리 함수를 학습할`
- `정상으로부터의 편차를 통해 이상 현상을 인식할 수 있음`

A noise image is generated by `a Perlin noise` generator to capture a variety of anomaly shapes (Figure 4, P ) and binarized by `a threshold sampled uniformly at random` (Figure 4, Ma) into an anomaly map $M_a$.

<aside>
💡 다양한 anomaly shape을 포착하기 위해 Perlin noise generator 사용

$M_a$ 는 무작위로 균일하게 샘플링된 threshold 로 인해 이진화됨

</aside>

The anomaly texture source image A is sampled from an anomaly source image dataset which is `unrelated to the input image distribution`

 {*posterize, sharpness, solarize, equalize, brightness change, color change, auto-contrast*} 중 3개가 랜덤으로 적용되어 Augmentation 된다.

이렇게 Augmented 된 texture image $A$는 the anomaly map $M_a$ 에 마스킹 된 후 $I$ 위에 합성된다. <br>
= $I_a$
![Untitled 9](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/dcac682c-25e6-4362-8992-73bbe278493f)

| $\overline{M}_a$ | inverse of $M_a$ |
| --- | --- |
| $\odot$ | the element-wise multiplication operation  |
| $\beta$ | the opacity parameter in blending.<br> sampled uniforms from an interval , $i.e., \beta \in [0.1, 1.0]$ |

## 3.4 Surface anomaly localization and detection
![Untitled 10](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/5c30e04a-6879-4123-bf91-2239704b0b0a)


1. **Local Average Pooling**
    
    $M_o$ is smoothed by a mean filter convolution layer `to aggregate the local anomaly response information.`
    
2. **Global max pooling**
    
    
3. **Compute anomaly score map**
    
    The final image-level anomaly score $\eta$ is computed by taking the maximum value of the smoothed anomaly score map:
    ![Untitled 11](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/2768cfd8-248a-4c04-bf12-a2153c96d65f)

        
    | $f_{(s_f \times s_f)}$ | a mean filter of size $s_f \times s_f$ <br> = Local Average Pooling 레이어의 필터 크기 |
    | --- | --- |
    | $*$ | the convolution operator  |
        
    ```python
    out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                     padding=21 // 2).cpu().detach().numpy()
    image_score = np.max(out_mask_averaged)
    
    anomaly_score_prediction.append(image_score)
    ```
    
    Threshold 는 우리가 정의해야 함
    

<aside>
💡 여기까지 Architecture 를 간단하게 정리하면

1. Making artificial anomalies image
    - a Perlin noise
    - binarized by a threshold sampled uniformly at random
    
2. The reconstructive Sub network
    
    인공적으로 만든 anomalies image를 anomaly-free image로 복원시키도록 학습
    
3. The discriminative sub-network
    
    anomaly detection mask, $M_o$ 를 출력하도록 학습
    
4. Surface anomaly localization and detection
    
    discriminative network에서 출력된 $M_o$ 가 2개의 Layer 거친 후 anomaly score 산출
    
</aside>

# 4. Experiments
![Untitled 12](https://github.com/CodeofO/DRAEM_BTS/assets/99871109/829782f8-f548-421f-9e3f-a6fa5937c4c1)

Figure 8. Qualitative examples. The original image, the anomaly map overlay, the anomaly map and the ground truth map are shown.
