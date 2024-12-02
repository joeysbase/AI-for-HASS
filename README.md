# AI-for-HASS
This repo provides various resources regarding computational aesthetic and artistic image/video generation, including papers, codes, and open-source datasets.
## Computational Aesthetic
### Papers&Codes
- **Attention-based Multi-Patch Aggregation for Image Aesthetic Assessment** [[paper](http://chongyangma.com/publications/am/2018_am_paper.pdf)] [[code](https://github.com/Openning07/MPADA)]
  
  Aggregation structures with explicit information, such as image attributes and scene semantics, are effective and popular for intelligent systems for assessing aesthetics of visual data. However, useful information may not be available due to the high cost of manual annotation and expert design. In this paper, we present a novel multi-patch (MP) aggregation method for image aesthetic assessment. Different from state-of-the-art methods, which augment an MP aggregation network with various visual attributes, we train the model in an end-to-end manner with aesthetic labels only (i.e., aesthetically positive or negative). We achieve the goal by resorting to an attention-based mechanism that adaptively adjusts the weight of each patch during the training process to improve learning efficiency. In addition, we propose a set of objectives with three typical attention mechanisms (i.e., average, minimum, and adaptive) and evaluate their effectiveness on the Aesthetic Visual Analysis (AVA) benchmark. Numerical results show that our approach outperforms existing methods by a large margin. We further verify the effectiveness of the proposed attention-based objectives via ablation studies and shed light on the design of aesthetic assessment systems.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

- **Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives** [[paper](https://arxiv.org/pdf/2211.04894v3)] [[code](https://github.com/vqassessment/dover)]
  
  The rapid increase in user-generated-content (UGC) videos calls for the development of effective video quality assessment (VQA) algorithms. However, the objective of the UGC-VQA problem is still ambiguous and can be viewed from two perspectives: the technical perspective, measuring the perception of distortions; and the aesthetic perspective, which relates to preference and recommendation on contents. To understand how these two perspectives affect overall subjective opinions in UGC-VQA, we conduct a large-scale subjective study to collect human quality opinions on overall quality of videos as well as perceptions from aesthetic and technical perspectives. The collected Disentangled Video Quality Database (DIVIDE-3k) confirms that human quality opinions on UGC videos are universally and inevitably affected by both aesthetic and technical perspectives. In light of this, we propose the Disentangled Objective Video Quality Evaluator (DOVER) to learn the quality of UGC videos based on the two perspectives. The DOVER proves state-of-the-art performance in UGC-VQA under very high efficiency. With perspective opinions in DIVIDE-3k, we further propose DOVER++, the first approach to provide reliable clear-cut quality evaluations from a single aesthetic or technical perspective. Code at https://github.com/VQAssessment/DOVER.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>
  
### Datasets
- **Gourmet Photography Dataset for Aesthetic Assessmentof Food Images** [[paper](https://www.researchgate.net/publication/329329757_Gourmet_photography_dataset_for_aesthetic_assessment_of_food_images)] [[code](https://github.com/Openning07/GPA)]
  
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

- **Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning** [[paper](https://arxiv.org/abs/2205.09542)] [[code](https://github.com/zyxElsa/CAST_pytorch)]
  
  In this work, we tackle the challenging problem of arbitrary image style transfer using a novel style feature representation learning method. A suitable style representation, as a key component in image stylization tasks, is essential to achieve satisfactory results. Existing deep neural network based approaches achieve reasonable results with the guidance from second-order statistics such as Gram matrix of content features. However, they do not leverage sufficient style information, which results in artifacts such as local distortions and style inconsistency. To address these issues, we propose to learn style representation directly from image features instead of their second-order statistics, by analyzing the similarities and differences between multiple styles and considering the style distribution. Specifically, we present Contrastive Arbitrary Style Transfer (CAST), which is a new style representation learning and style transfer method via contrastive learning. Our framework consists of three key components, i.e., a multi-layer style projector for style code encoding, a domain enhancement module for effective learning of style distribution, and a generative network for image style transfer. We conduct qualitative and quantitative evaluations comprehensively to demonstrate that our approach achieves significantly better results compared to those obtained via state-of-the-art methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

- **$StyTr^2$ : Image Style Transfer with Transformers** [[paper](https://arxiv.org/abs/2105.14576)] [[code](https://github.com/diyiiyiii/StyTR-2)]
  
  The goal of image style transfer is to render an image with artistic features guided by a style reference while maintaining the original content. Owing to the locality in convolutional neural networks (CNNs), extracting and maintaining the global information of input images is difficult. Therefore, traditional neural style transfer methods face biased content representation. To address this critical issue, we take long-range dependencies of input images into account for image style transfer by proposing a transformer-based approach called StyTr2. In contrast with visual transformers for other vision tasks, StyTr2 contains two different transformer encoders to generate domain-specific sequences for content and style, respectively. Following the encoders, a multi-layer transformer decoder is adopted to stylize the content sequence according to the style sequence. We also analyze the deficiency of existing positional encoding methods and propose the content-aware positional encoding (CAPE), which is scale-invariant and more suitable for image style transfer tasks. Qualitative and quantitative experiments demonstrate the effectiveness of the proposed StyTr2 compared with state-of-the-art CNN-based and flow-based approaches.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

- **Arbitrary Style Transfer via Multi-Adaptation Network** [[paper](https://arxiv.org/abs/2005.13219)] [[code](https://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/)]
  
  Arbitrary style transfer is a significant topic with research value and application prospect. A desired style transfer, given a content image and referenced style painting, would render the content image with the color tone and vivid stroke patterns of the style painting while synchronously maintaining the detailed content structure information. Style transfer approaches would initially learn content and style representations of the content and style references and then generate the stylized images guided by these representations. In this paper, we propose the multi-adaptation network which involves two self-adaptation (SA) modules and one co-adaptation (CA) module: the SA modules adaptively disentangle the content and style representations, i.e., content SA module uses position-wise self-attention to enhance content representation and style SA module uses channel-wise self-attention to enhance style representation; the CA module rearranges the distribution of style representation based on content representation distribution by calculating the local similarity between the disentangled content and style features in a non-local fashion. Moreover, a new disentanglement loss function enables our network to extract main style patterns and exact content structures to adapt to various input images, respectively. Various qualitative and quantitative experiments demonstrate that the proposed multi-adaptation network leads to better results than the state-of-the-art style transfer methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

- **Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion** [[paper](https://arxiv.org/abs/2209.13360)] [[code](https://github.com/haha-lisa/MGAD-multimodal-guided-artwork-diffusion)]
  
  Digital art creation is getting more attention in the multimedia community for providing effective engagement of the public with art. Current digital art generation methods usually use single modality inputs as guidance, limiting the expressiveness of the model and the diversity of generated results. To solve this problem, we propose the multimodal guided artwork diffusion (MGAD) model, a diffusion-based digital artwork generation method that utilizes multimodal prompts as guidance to control the classifier-free diffusion model. Additionally, the contrastive language-image pretraining (CLIP) model is used to unify text and image modalities. However, the semantic content of multimodal prompts may conflict with each other, which leads to a collapse in generating progress. Extensive experimental results on the quality and quantity of the generated digital art paintings confirm the effectiveness of the combination of the diffusion model and multimodal guidance.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

- **SAN: INDUCING METRIZABILITY OF GAN WITH DISCRIMINATIVE NORMALIZED LINEAR LAYER** [[paper](https://arxiv.org/pdf/2301.12811v4)] [[code](https://github.com/sony/san)]
  
  Generative adversarial networks (GANs) learn a target probability distribution by optimizing a generator and a discriminator with minimax objectives. This paper addresses the question of whether such optimization actually provides the generator with gradients that make its distribution close to the target distribution. We derive metrizable conditions, sufficient conditions for the discriminator to serve as the distance between the distributions, by connecting the GAN formulation with the concept of sliced optimal transport. Furthermore, by leveraging these theoretical results, we propose a novel GAN training scheme called the Slicing Adversarial Network (SAN). With only simple modifications, a broad class of existing GANs can be converted to SANs. Experiments on synthetic and image datasets support our theoretical results and the effectiveness of SAN as compared to the usual GANs. We also apply SAN to StyleGAN-XL, which leads to a state-of-the-art FID score amongst GANs for class conditional generation on CIFAR10 and ImageNet 256$times$256.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

- **A Neural Algorithm of Artistic Style** [[paper](https://arxiv.org/pdf/1508.06576v2)] [[code](https://github.com/jcjohnson/neural-style)]
  
  In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image. Thus far the algorithmic basis of this process is unknown and there exists no artificial system with similar capabilities. However, in other key areas of visual perception such as object and face recognition near-human performance was recently demonstrated by a class of biologically inspired vision models called Deep Neural Networks. Here we introduce an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images. Moreover, in light of the striking similarities between performance-optimised artificial neural networks and biological vision, our work offers a path forward to an algorithmic understanding of how humans create and perceive artistic imagery.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

- **High-Resolution Image Synthesis with Latent Diffusion Models** [[paper](https://arxiv.org/pdf/2112.10752v2)] [[code](https://github.com/CompVis/latent-diffusion)]
  
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

- **Conditional Image Synthesis with Auxiliary Classifier GANs** [[paper](https://arxiv.org/pdf/1610.09585v4)] [[code](https://github.com/dacostaHugo/AC-GANs)]
  
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

- **LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS** [[paper](https://arxiv.org/pdf/1809.11096v2)] [[code](https://github.com/taki0112/BigGAN-Tensorflow)]
  
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

- **Semantic Image Synthesis with Spatially-Adaptive Normalization** [[paper](https://arxiv.org/pdf/1903.07291v2)] [[code](https://github.com/NVlabs/SPADE)]
  
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

- **StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks** [[paper](https://arxiv.org/pdf/1710.10916v3)] [[code](https://github.com/hanzhanggit/StackGAN)]
  
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
- **FFHQ** [[paper](https://arxiv.org/pdf/1812.04948v3)] [[code](https://github.com/NVlabs/ffhq-dataset)]
  
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

- **Arbitrary Video Style Transfer via Multi-Channel Correlation** [[paper](https://arxiv.org/abs/2009.08003)] [[code](https://github.com/diyiiyiii/MCCNet)]
  
  Video style transfer is getting more attention in AI community for its numerous applications such as augmented reality and animation productions. Compared with traditional image style transfer, performing this task on video presents new challenges: how to effectively generate satisfactory stylized results for any specified style, and maintain temporal coherence across frames at the same time. Towards this end, we propose Multi-Channel Correction network (MCCNet), which can be trained to fuse the exemplar style features and input content features for efficient style transfer while naturally maintaining the coherence of input videos. Specifically, MCCNet works directly on the feature space of style and content domain where it learns to rearrange and fuse style features based on their similarity with content features. The outputs generated by MCC are features containing the desired style patterns which can further be decoded into images with vivid style textures. Moreover, MCCNet is also designed to explicitly align the features to input which ensures the output maintains the content structures as well as the temporal continuity. To further improve the performance of MCCNet under complex light conditions, we also introduce the illumination loss during training. Qualitative and quantitative evaluations demonstrate that MCCNet performs well in both arbitrary video and image style transfer tasks.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

21. **Everybody Dance Now** [[paper](https://arxiv.org/pdf/1808.07371v2)] [[code](https://github.com/carolineec/EverybodyDanceNow)]
  
  This paper presents a simple method for "do as I do" motion transfer: given a source video of a person dancing, we can transfer that performance to a novel (amateur) target after only a few minutes of the target subject performing standard moves. We approach this problem as video-to-video translation using pose as an intermediate representation. To transfer the motion, we extract poses from the source subject and apply the learned pose-to-appearance mapping to generate the target subject. We predict two consecutive frames for temporally coherent video results and introduce a separate pipeline for realistic face synthesis. Although our method is quite simple, it produces surprisingly compelling results (see video). This motivates us to also provide a forensics tool for reliable synthetic content detection, which is able to distinguish videos synthesized by our system from real data. In addition, we release a first-of-its-kind open-source dataset of videos that can be legally used for training and motion transfer.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

22. **Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation** [[paper](https://arxiv.org/pdf/1811.09393v4)] [[code](https://github.com/thunil/TecoGAN)]
  
  Our work explores temporal self-supervision for GAN-based video generation tasks. While adversarial training successfully yields generative models for a variety of areas, temporal relationships in the generated data are much less explored. Natural temporal changes are crucial for sequential generation tasks, e.g. video super-resolution and unpaired video translation. For the former, state-of-the-art methods often favor simpler norm losses such as 
 over adversarial training. However, their averaging nature easily leads to temporally smooth results with an undesirable lack of spatial detail. For unpaired video translation, existing approaches modify the generator networks to form spatio-temporal cycle consistencies. In contrast, we focus on improving learning objectives and propose a temporally self-supervised algorithm. For both tasks, we show that temporal adversarial learning is key to achieving temporally coherent solutions without sacrificing spatial detail. We also propose a novel Ping-Pong loss to improve the long-term temporal consistency. It effectively prevents recurrent networks from accumulating artifacts temporally without depressing detailed features. Additionally, we propose a first set of metrics to quantitatively evaluate the accuracy as well as the perceptual quality of the temporal evolution. A series of user studies confirm the rankings computed with these metrics. Code, data, models, and results are provided at https://github.com/thunil/TecoGAN. The project page https://ge.in.tum.de/publications/2019-tecogan-chu/ contains supplemental materials.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

23. **Artistic style transfer for videos** [[paper](https://arxiv.org/pdf/1604.08610v2)] [[code](https://github.com/manuelruder/artistic-videos)]
  
  In the past, manually re-drawing an image in a certain artistic style required a professional artist and a long time. Doing this for a video sequence single-handed was beyond imagination. Nowadays computers provide new possibilities. We present an approach that transfers the style from one image (for example, a painting) to a whole video sequence. We make use of recent advances in style transfer in still images and propose new initializations and loss functions applicable to videos. This allows us to generate consistent and stable stylized video sequences, even in cases with large motion and strong occlusion. We show that the proposed method clearly outperforms simpler baselines both qualitatively and quantitatively.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

24. **Artistic style transfer for videos and spherical images** [[paper](https://arxiv.org/pdf/1708.04538v3)] [[code](https://github.com/manuelruder/fast-artistic-videos)]
  
  Manually re-drawing an image in a certain artistic style takes a professional artist a long time. Doing this for a video sequence single-handedly is beyond imagination. We present two computational approaches that transfer the style from one image (for example, a painting) to a whole video sequence. In our first approach, we adapt to videos the original image style transfer technique by Gatys et al. based on energy minimization. We introduce new ways of initialization and new loss functions to generate consistent and stable stylized video sequences even in cases with large motion and strong occlusion. Our second approach formulates video stylization as a learning problem. We propose a deep network architecture and training procedures that allow us to stylize arbitrary-length videos in a consistent and stable way, and nearly in real time. We show that the proposed methods clearly outperform simpler baselines both qualitatively and quantitatively. Finally, we propose a way to adapt these approaches also to 360 degree images and videos as they emerge with recent virtual reality hardware.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

25. **Music2Video: Automatic Generation of Music Video with fusion of audio and text** [[paper](https://arxiv.org/pdf/2201.03809v2)] [[code](https://github.com/joeljang/music2video)]
  
  Creation of images using generative adversarial networks has been widely adapted into multi-modal regime with the advent of multi-modal representation models pre-trained on large corpus. Various modalities sharing a common representation space could be utilized to guide the generative models to create images from text or even from audio source. Departing from the previous methods that solely rely on either text or audio, we exploit the expressiveness of both modality. Based on the fusion of text and audio, we create video whose content is consistent with the distinct modalities that are provided. A simple approach to automatically segment the video into variable length intervals and maintain time consistency in generated video is part of our method. Our proposed framework for generating music video shows promising results in application level where users can interactively feed in music source and text source to create artistic music videos. Our code is available at https://github.com/joeljang/music2video.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

26. **Video Diffusion Models** [[paper](https://arxiv.org/pdf/2204.03458v2)] [[code](https://github.com/lucidrains/video-diffusion-pytorch)]
  
  Generating temporally coherent high fidelity video is an important milestone in generative modeling research. We make progress towards this milestone by proposing a diffusion model for video generation that shows very promising initial results. Our model is a natural extension of the standard image diffusion architecture, and it enables jointly training from image and video data, which we find to reduce the variance of minibatch gradients and speed up optimization. To generate long and higher resolution videos we introduce a new conditional sampling technique for spatial and temporal video extension that performs better than previously proposed methods. We present the first results on a large text-conditioned video generation task, as well as state-of-the-art results on established benchmarks for video prediction and unconditional video generation.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>



28. **Consistent Video Style Transfer via Relaxation and Regularization** [[paper](https://ieeexplore.ieee.org/document/9204808)] [[code](https://github.com/daooshee/ReReVST-Code)]
  
  In recent years, neural style transfer has attracted more and more attention, especially for image style transfer. However, temporally consistent style transfer for videos is still a challenging problem. Existing methods, either relying on a significant amount of video data with optical flows or using single-frame regularizers, fail to handle strong motions or complex variations, therefore have limited performance on real videos. In this article, we address the problem by jointly considering the intrinsic properties of stylization and temporal consistency. We first identify the cause of the conflict between style transfer and temporal consistency, and propose to reconcile this contradiction by relaxing the objective function, so as to make the stylization loss term more robust to motions. Through relaxation, style transfer is more robust to inter-frame variation without degrading the subjective effect. Then, we provide a novel formulation and understanding of temporal consistency. Based on the formulation, we analyze the drawbacks of existing training strategies and derive a new regularization. We show by experiments that the proposed regularization can better balance the spatial and temporal performance. Based on relaxation and regularization, we design a zero-shot video style transfer framework. Moreover, for better feature migration, we introduce a new module to dynamically adjust inter-channel distributions. Quantitative and qualitative results demonstrate the superiority of our method over state-of-the-art style transfer methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

29. **DwNet: Dense warp-based network for pose-guided human video generation** [[paper](https://arxiv.org/pdf/1910.09139v1)] [[code](https://github.com/ubc-vision/DwNet)]
  
  Generation of realistic high-resolution videos of human subjects is a challenging and important task in computer vision. In this paper, we focus on human motion transfer - generation of a video depicting a particular subject, observed in a single image, performing a series of motions exemplified by an auxiliary (driving) video. Our GAN-based architecture, DwNet, leverages dense intermediate pose-guided representation and refinement process to warp the required subject appearance, in the form of the texture, from a source image into a desired pose. Temporal consistency is maintained by further conditioning the decoding process within a GAN on the previously generated frame. In this way a video is generated in an iterative and recurrent fashion. We illustrate the efficacy of our approach by showing state-of-the-art quantitative and qualitative performance on two benchmark datasets: TaiChi and Fashion Modeling. The latter is collected by us and will be made publicly available to the community.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

30. **ReCoNet: Real-time Coherent Video Style Transfer Network** [[paper](https://arxiv.org/pdf/1807.01197v2)] [[code](https://github.com/safwankdb/ReCoNet-PyTorch)]
  
  Image style transfer models based on convolutional neural networks usually suffer from high temporal inconsistency when applied to videos. Some video style transfer models have been proposed to improve temporal consistency, yet they fail to guarantee fast processing speed, nice perceptual style quality and high temporal consistency at the same time. In this paper, we propose a novel real-time video style transfer model, ReCoNet, which can generate temporally coherent style transfer videos while maintaining favorable perceptual styles. A novel luminance warping constraint is added to the temporal loss at the output level to capture luminance changes between consecutive frames and increase stylization stability under illumination effects. We also propose a novel feature-map-level temporal loss to further enhance temporal consistency on traceable objects. Experimental results indicate that our model exhibits outstanding performance both qualitatively and quantitatively.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

### Datasets
- **ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation** [[paper](https://arxiv.org/pdf/2406.18522v2)] [[code](https://github.com/pku-yuangroup/chronomagic-bench)]
  
  We propose a novel text-to-video (T2V) generation benchmark, ChronoMagic-Bench, to evaluate the temporal and metamorphic capabilities of the T2V models (e.g. Sora and Lumiere) in time-lapse video generation. In contrast to existing benchmarks that focus on visual quality and textual relevance of generated videos, ChronoMagic-Bench focuses on the model's ability to generate time-lapse videos with significant metamorphic amplitude and temporal coherence. The benchmark probes T2V models for their physics, biology, and chemistry capabilities, in a free-form text query. For these purposes, ChronoMagic-Bench introduces 1,649 prompts and real-world videos as references, categorized into four major types of time-lapse videos: biological, human-created, meteorological, and physical phenomena, which are further divided into 75 subcategories. This categorization comprehensively evaluates the model's capacity to handle diverse and complex transformations. To accurately align human preference with the benchmark, we introduce two new automatic metrics, MTScore and CHScore, to evaluate the videos' metamorphic attributes and temporal coherence. MTScore measures the metamorphic amplitude, reflecting the degree of change over time, while CHScore assesses the temporal coherence, ensuring the generated videos maintain logical progression and continuity. Based on ChronoMagic-Bench, we conduct comprehensive manual evaluations of ten representative T2V models, revealing their strengths and weaknesses across different categories of prompts, and providing a thorough evaluation framework that addresses current gaps in video generation research. Moreover, we create a large-scale ChronoMagic-Pro dataset, containing 460k high-quality pairs of 720p time-lapse videos and detailed captions ensuring high physical pertinence and large metamorphic amplitude. [Homepage](https://pku-yuangroup.github.io/ChronoMagic-Bench/).
  
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

