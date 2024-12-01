# AI-for-HASS
This repo provides various resources regarding computational aesthetic and artistic image/video generation, including papers, codes, and open-source datasets.
## Computational Aesthetic
### Papers&Codes
12. **Attention-based Multi-Patch Aggregation for Image Aesthetic Assessment** [[paper](http://chongyangma.com/publications/am/2018_am_paper.pdf)] [[code](https://github.com/Openning07/MPADA)]
  
  Aggregation structures with explicit information, such as image attributes and scene semantics, are effective and popular for intelligent systems for assessing aesthetics of visual data. However, useful information may not be available due to the high cost of manual annotation and expert design. In this paper, we present a novel multi-patch (MP) aggregation method for image aesthetic assessment. Different from state-of-the-art methods, which augment an MP aggregation network with various visual attributes, we train the model in an end-to-end manner with aesthetic labels only (i.e., aesthetically positive or negative). We achieve the goal by resorting to an attention-based mechanism that adaptively adjusts the weight of each patch during the training process to improve learning efficiency. In addition, we propose a set of objectives with three typical attention mechanisms (i.e., average, minimum, and adaptive) and evaluate their effectiveness on the Aesthetic Visual Analysis (AVA) benchmark. Numerical results show that our approach outperforms existing methods by a large margin. We further verify the effectiveness of the proposed attention-based objectives via ablation studies and shed light on the design of aesthetic assessment systems.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>
  
### Datasets
11. **Gourmet Photography Dataset for Aesthetic Assessmentof Food Images** [[paper](https://www.researchgate.net/publication/329329757_Gourmet_photography_dataset_for_aesthetic_assessment_of_food_images)] [[code](https://github.com/Openning07/GPA)]
  
  In this study, we present the Gourmet Photography Dataset (GPD),which is the rst large-scale dataset for aesthetic assessment offood photographs. We collect 12,000 food images together withhuman-annotated labels (i.e., aesthetically positive or negative) tobuild this dataset. We evaluate the performance of several popu-lar machine learning algorithms for aesthetic assessment of foodimages to verify the eectiveness and importance of our GPDdataset. Experimental results show that deep convolutional neuralnetworks trained on GPD can achieve comparable performancewith human experts in this task, even on unseen food photographs.Our experiments also provide insights to support further study andapplications related to visual analysis of food images
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

## Artistic Image Generation
### Papers&Codes
- **$Z^*$ : Zero-shot Style Transfer via Attention Rearrangement** [[paper](https://arxiv.org/abs/2311.16491)] [[code](https://github.com/HolmesShuan/Zero-shot-Style-Transfer-via-Attention-Rearrangement)]
  
  Z-STAR is an innovative zero-shot (training-free) style transfer method that leverages the generative prior knowledge within a pre-trained diffusion model. By employing an attention rearrangement strategy, it effectively fuses content and style information without the need for retraining or tuning for each input style.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/zstar.png" width="600">
  </p>

- **A Unified Arbitrary Style Transfer Framework via Adaptive Contrastive Learning** [[paper](https://dl.acm.org/doi/pdf/10.1145/3605548)] [[code](https://github.com/zyxElsa/CAST_pytorch)]
  
  In this work, we tackle the challenging problem of arbitrary image style transfer using a novel style feature representation learning method. A suitable style representation, as a key component in image stylization tasks, is essential to achieve satisfactory results. Existing deep neural network based approaches achieve reasonable results with the guidance from second-order statistics such as Gram matrix of content features. However, they do not leverage sufficient style information, which results in artifacts such as local distortions and style inconsistency. To address these issues, we propose to learn style representation directly from image features instead of their second-order statistics, by analyzing the similarities and differences between multiple styles and considering the style distribution.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/sav.png" width="600">
  </p>

- **ProSpect: Prompt Spectrum for Attribute-Aware Personalization of Diffusion Models** [[paper](https://arxiv.org/pdf/2305.16225)] [[code](https://github.com/zyxElsa/ProSpect)]
  
  Personalizing generative models offers a way to guide image generation with user-provided references. Current personalization methods can invert an object or concept into the textual conditioning space and compose new natural sentences for text-to-image diffusion models. However, representing and editing specific visual attributes like material, style, layout, etc. remains a challenge, leading to a lack of disentanglement and editability. To address this, we propose a novel approach that leverages the step-by-step generation process of diffusion models, which generate images from low- to high-frequency information, providing a new perspective on representing, generating, and editing images. We develop Prompt Spectrum Space P*, an expanded textual conditioning space, and a new image representation method called ProSpect. ProSpect represents an image as a collection of inverted textual token embeddings encoded from per-stage prompts, where each prompt corresponds to a specific generation stage (i.e., a group of consecutive steps) of the diffusion model. Experimental results demonstrate that P* and ProSpect offer stronger disentanglement and controllability compared to existing methods. We apply ProSpect in various personalized attribute-aware image generation applications, such as image/text-guided material/style/layout transfer/editing, achieving previously unattainable results with a single image input without fine-tuning the diffusion models.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/sav.png" width="600">
  </p>

- **Inversion-Based Style Transfer with Diffusion Models** [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Inversion-Based_Style_Transfer_With_Diffusion_Models_CVPR_2023_paper.pdf)] [[code](https://github.com/zyxElsa/InST)]
  
  The artistic style within a painting is the means of expression, which includes not only the painting material, colors, and brushstrokes, but also the high-level attributes including semantic elements, object shapes, etc. Previous arbitrary example-guided artistic image generation methods often fail to control shape changes or convey elements. The pre-trained text-to-image synthesis diffusion probabilistic models have achieved remarkable quality, but it often requires extensive textual descriptions to accurately portray attributes of a particular painting. We believe that the uniqueness of an artwork lies precisely in the fact that it cannot be adequately explained with normal language.Our key idea is to learn artistic style directly from a single painting and then guide the synthesis without providing complex textual descriptions. Specifically, we assume style as a learnable textual description of a painting. We propose an inversion-based style transfer method (InST), which can efficiently and accurately learn the key information of an image, thus capturing and transferring the complete artistic style of a painting. We demonstrate the quality and efficiency of our method on numerous paintings of various artists and styles.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

7. **Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning** [[paper](https://arxiv.org/abs/2205.09542)] [[code](https://github.com/zyxElsa/CAST_pytorch)]
  
  In this work, we tackle the challenging problem of arbitrary image style transfer using a novel style feature representation learning method. A suitable style representation, as a key component in image stylization tasks, is essential to achieve satisfactory results. Existing deep neural network based approaches achieve reasonable results with the guidance from second-order statistics such as Gram matrix of content features. However, they do not leverage sufficient style information, which results in artifacts such as local distortions and style inconsistency. To address these issues, we propose to learn style representation directly from image features instead of their second-order statistics, by analyzing the similarities and differences between multiple styles and considering the style distribution. Specifically, we present Contrastive Arbitrary Style Transfer (CAST), which is a new style representation learning and style transfer method via contrastive learning. Our framework consists of three key components, i.e., a multi-layer style projector for style code encoding, a domain enhancement module for effective learning of style distribution, and a generative network for image style transfer. We conduct qualitative and quantitative evaluations comprehensively to demonstrate that our approach achieves significantly better results compared to those obtained via state-of-the-art methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

8. **$StyTr^2$ : Image Style Transfer with Transformers** [[paper](https://arxiv.org/abs/2105.14576)] [[code](https://github.com/diyiiyiii/StyTR-2)]
  
  The goal of image style transfer is to render an image with artistic features guided by a style reference while maintaining the original content. Owing to the locality in convolutional neural networks (CNNs), extracting and maintaining the global information of input images is difficult. Therefore, traditional neural style transfer methods face biased content representation. To address this critical issue, we take long-range dependencies of input images into account for image style transfer by proposing a transformer-based approach called StyTr2. In contrast with visual transformers for other vision tasks, StyTr2 contains two different transformer encoders to generate domain-specific sequences for content and style, respectively. Following the encoders, a multi-layer transformer decoder is adopted to stylize the content sequence according to the style sequence. We also analyze the deficiency of existing positional encoding methods and propose the content-aware positional encoding (CAPE), which is scale-invariant and more suitable for image style transfer tasks. Qualitative and quantitative experiments demonstrate the effectiveness of the proposed StyTr2 compared with state-of-the-art CNN-based and flow-based approaches.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

10. **Arbitrary Style Transfer via Multi-Adaptation Network** [[paper](https://arxiv.org/abs/2005.13219)] [[code](https://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/)]
  
  Arbitrary style transfer is a significant topic with research value and application prospect. A desired style transfer, given a content image and referenced style painting, would render the content image with the color tone and vivid stroke patterns of the style painting while synchronously maintaining the detailed content structure information. Style transfer approaches would initially learn content and style representations of the content and style references and then generate the stylized images guided by these representations. In this paper, we propose the multi-adaptation network which involves two self-adaptation (SA) modules and one co-adaptation (CA) module: the SA modules adaptively disentangle the content and style representations, i.e., content SA module uses position-wise self-attention to enhance content representation and style SA module uses channel-wise self-attention to enhance style representation; the CA module rearranges the distribution of style representation based on content representation distribution by calculating the local similarity between the disentangled content and style features in a non-local fashion. Moreover, a new disentanglement loss function enables our network to extract main style patterns and exact content structures to adapt to various input images, respectively. Various qualitative and quantitative experiments demonstrate that the proposed multi-adaptation network leads to better results than the state-of-the-art style transfer methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

6. **Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion** [[paper](https://arxiv.org/abs/2209.13360)] [[code](https://github.com/haha-lisa/MGAD-multimodal-guided-artwork-diffusion)]
  
  Digital art creation is getting more attention in the multimedia community for providing effective engagement of the public with art. Current digital art generation methods usually use single modality inputs as guidance, limiting the expressiveness of the model and the diversity of generated results. To solve this problem, we propose the multimodal guided artwork diffusion (MGAD) model, a diffusion-based digital artwork generation method that utilizes multimodal prompts as guidance to control the classifier-free diffusion model. Additionally, the contrastive language-image pretraining (CLIP) model is used to unify text and image modalities. However, the semantic content of multimodal prompts may conflict with each other, which leads to a collapse in generating progress. Extensive experimental results on the quality and quantity of the generated digital art paintings confirm the effectiveness of the combination of the diffusion model and multimodal guidance.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

13. **SAN: INDUCING METRIZABILITY OF GAN WITH DISCRIMINATIVE NORMALIZED LINEAR LAYER** [[paper](https://arxiv.org/pdf/2301.12811v4)] [[code](https://github.com/sony/san)]
  
  Generative adversarial networks (GANs) learn a target probability distribution by optimizing a generator and a discriminator with minimax objectives. This paper addresses the question of whether such optimization actually provides the generator with gradients that make its distribution close to the target distribution. We derive metrizable conditions, sufficient conditions for the discriminator to serve as the distance between the distributions, by connecting the GAN formulation with the concept of sliced optimal transport. Furthermore, by leveraging these theoretical results, we propose a novel GAN training scheme called the Slicing Adversarial Network (SAN). With only simple modifications, a broad class of existing GANs can be converted to SANs. Experiments on synthetic and image datasets support our theoretical results and the effectiveness of SAN as compared to the usual GANs. We also apply SAN to StyleGAN-XL, which leads to a state-of-the-art FID score amongst GANs for class conditional generation on CIFAR10 and ImageNet 256$times$256.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

15. **Generative Adversarial Text to Image Synthesis** [[paper](https://arxiv.org/pdf/1605.05396v2)] [[code](https://github.com/reedscot/icml2016)]
  
  Automatic synthesis of realistic images from text would be interesting and useful, but current AI systems are still far from this goal. However, in recent years generic and powerful recurrent neural network architectures have been developed to learn discriminative text feature representations. Meanwhile, deep convolutional generative adversarial networks (GANs) have begun to generate highly compelling images of specific categories, such as faces, album covers, and room interiors. In this work, we develop a novel deep architecture and GAN formulation to effectively bridge these advances in text and image modeling, translating visual concepts from characters to pixels. We demonstrate the capability of our model to generate plausible images of birds and flowers from detailed text descriptions.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

16. **High-Resolution Image Synthesis with Latent Diffusion Models** [[paper](https://arxiv.org/pdf/2112.10752v2)] [[code](https://github.com/CompVis/latent-diffusion)]
  
  By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion
models (DMs) achieve state-of-the-art synthesis results on
image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these
models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU
days and inference is expensive due to sequential evaluations. To enable DM training on limited computational
resources while retaining their quality and flexibility, we
apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion
models on such a representation allows for the first time
to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity.
By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text
or bounding boxes and high-resolution synthesis becomes
possible in a convolutional manner. Our latent diffusion
models (LDMs) achieve new state-of-the-art scores for image inpainting and class-conditional image synthesis and
highly competitive performance on various tasks, including text-to-image synthesis, unconditional image generation
and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

17. **Conditional Image Synthesis with Auxiliary Classifier GANs** [[paper](https://arxiv.org/pdf/1610.09585v4)] [[code](https://github.com/dacostaHugo/AC-GANs)]
  
  In this paper we introduce new methods for the
improved training of generative adversarial networks (GANs) for image synthesis. We construct a variant of GANs employing label conditioning that results in 128 × 128 resolution image samples exhibiting global coherence. We
expand on previous work for image quality assessment to provide two new analyses for assessing the discriminability and diversity of samples
from class-conditional image synthesis models.
These analyses demonstrate that high resolution
samples provide class information not present in
low resolution samples. Across 1000 ImageNet
classes, 128 × 128 samples are more than twice
as discriminable as artificially resized 32 × 32
samples. In addition, 84.7% of the classes have
samples exhibiting diversity comparable to real
ImageNet data.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

18. **LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS** [[paper](https://arxiv.org/pdf/1809.11096v2)] [[code](https://github.com/taki0112/BigGAN-Tensorflow)]
  
  Despite recent progress in generative image modeling, successfully generating
high-resolution, diverse samples from complex datasets such as ImageNet remains
an elusive goal. To this end, we train Generative Adversarial Networks at the
largest scale yet attempted, and study the instabilities specific to such scale. We
find that applying orthogonal regularization to the generator renders it amenable
to a simple “truncation trick,” allowing fine control over the trade-off between
sample fidelity and variety by reducing the variance of the Generator’s input. Our
modifications lead to models which set the new state of the art in class-conditional
image synthesis. When trained on ImageNet at 128×128 resolution, our models
(BigGANs) achieve an Inception Score (IS) of 166.5 and Fr ́echet Inception Distance (FID) of 7.4, improving over the previous best IS of 52.52 and FID of 18.65.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

19. **Semantic Image Synthesis with Spatially-Adaptive Normalization** [[paper](https://arxiv.org/pdf/1903.07291v2)] [[code](https://github.com/NVlabs/SPADE)]
  
  We propose spatially-adaptive normalization, a simple
but effective layer for synthesizing photorealistic images
given an input semantic layout. Previous methods directly
feed the semantic layout as input to the deep network, which
is then processed through stacks of convolution, normalization, and nonlinearity layers. We show that this is suboptimal as the normalization layers tend to “wash away” semantic information. To address the issue, we propose using
the input layout for modulating the activations in normalization layers through a spatially-adaptive, learned transformation. Experiments on several challenging datasets
demonstrate the advantage of the proposed method over existing approaches, regarding both visual fidelity and alignment with input layouts. Finally, our model allows user
control over both semantic and style
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

20. **StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks** [[paper](https://arxiv.org/pdf/1710.10916v3)] [[code](https://github.com/hanzhanggit/StackGAN)]
  
  Although Generative Adversarial Networks (GANs) have shown remarkable success in various tasks, they still face
challenges in generating high quality images. In this paper, we propose Stacked Generative Adversarial Networks (StackGANs)
aimed at generating high-resolution photo-realistic images. First, we propose a two-stage generative adversarial network architecture,
StackGAN-v1, for text-to-image synthesis. The Stage-I GAN sketches the primitive shape and colors of a scene based on a given text
description, yielding low-resolution images. The Stage-II GAN takes Stage-I results and the text description as inputs, and generates
high-resolution images with photo-realistic details. Second, an advanced multi-stage generative adversarial network architecture,
StackGAN-v2, is proposed for both conditional and unconditional generative tasks. Our StackGAN-v2 consists of multiple generators
and multiple discriminators arranged in a tree-like structure; images at multiple scales corresponding to the same scene are generated
from different branches of the tree. StackGAN-v2 shows more stable training behavior than StackGAN-v1 by jointly approximating
multiple distributions. Extensive experiments demonstrate that the proposed stacked generative adversarial networks significantly
outperform other state-of-the-art methods in generating photo-realistic images.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

### Datasets
14. **FFHQ** [[paper](https://arxiv.org/pdf/1812.04948v3)] [[code](https://github.com/NVlabs/ffhq-dataset)]
  
  A dataset of human faces (Flickr-Faces-HQ, FFHQ) that offers much higher quality and covers considerably wider variation than existing high-resolution dataset
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>
  
## Artistic Video Generation
### Papers&Codes
- **Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer** [[paper](https://arxiv.org/pdf/2305.05464)] [[code](https://github.com/haha-lisa/Style-A-Video)]
  
  This paper proposes a zero-shot video stylization method named Style-A-Video, which utilizes a generative pre-trained transformer with an image latent diffusion model to achieve a concise text-controlled video stylization.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/sav.png" width="600">
  </p>

9. **Arbitrary Video Style Transfer via Multi-Channel Correlation** [[paper](https://arxiv.org/abs/2009.08003)] [[code](https://github.com/diyiiyiii/MCCNet)]
  
  Video style transfer is getting more attention in AI community for its numerous applications such as augmented reality and animation productions. Compared with traditional image style transfer, performing this task on video presents new challenges: how to effectively generate satisfactory stylized results for any specified style, and maintain temporal coherence across frames at the same time. Towards this end, we propose Multi-Channel Correction network (MCCNet), which can be trained to fuse the exemplar style features and input content features for efficient style transfer while naturally maintaining the coherence of input videos. Specifically, MCCNet works directly on the feature space of style and content domain where it learns to rearrange and fuse style features based on their similarity with content features. The outputs generated by MCC are features containing the desired style patterns which can further be decoded into images with vivid style textures. Moreover, MCCNet is also designed to explicitly align the features to input which ensures the output maintains the content structures as well as the temporal continuity. To further improve the performance of MCCNet under complex light conditions, we also introduce the illumination loss during training. Qualitative and quantitative evaluations demonstrate that MCCNet performs well in both arbitrary video and image style transfer tasks.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>
  
### Datasets












21. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

22. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

23. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

24. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

25. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

26. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

27. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

28. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

29. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

30. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

31. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

32. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

33. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

34. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

35. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

36. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

37. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

38. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

39. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

40. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

41. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

42. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

43. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

44. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

45. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

46. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

47. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

48. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

49. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

50. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

51. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

52. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

53. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

54. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

55. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

56. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

