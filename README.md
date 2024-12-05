# AI-for-HASS
This repo provides various resources regarding computational aesthetic and artistic image/video generation, including papers, codes, and open-source datasets.
## Computational Aesthetic
### Papers&Codes
- **Attention-based Multi-Patch Aggregation for Image Aesthetic Assessment** [[paper](http://chongyangma.com/publications/am/2018_am_paper.pdf)] [[code](https://github.com/Openning07/MPADA)]
  
  Aggregation structures with explicit information, such as image attributes and scene semantics, are effective and popular for intelligent systems for assessing aesthetics of visual data. However, useful information may not be available due to the high cost of manual annotation and expert design. In this paper, we present a novel multi-patch (MP) aggregation method for image aesthetic assessment. Different from state-of-the-art methods, which augment an MP aggregation network with various visual attributes, we train the model in an end-to-end manner with aesthetic labels only (i.e., aesthetically positive or negative). We achieve the goal by resorting to an attention-based mechanism that adaptively adjusts the weight of each patch during the training process to improve learning efficiency. In addition, we propose a set of objectives with three typical attention mechanisms (i.e., average, minimum, and adaptive) and evaluate their effectiveness on the Aesthetic Visual Analysis (AVA) benchmark. Numerical results show that our approach outperforms existing methods by a large margin. We further verify the effectiveness of the proposed attention-based objectives via ablation studies and shed light on the design of aesthetic assessment systems.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/1.jpg">
  </p>

- **Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives** [[paper](https://arxiv.org/pdf/2211.04894v3)] [[code](https://github.com/vqassessment/dover)]
  
  The rapid increase in user-generated-content (UGC) videos calls for the development of effective video quality assessment (VQA) algorithms. However, the objective of the UGC-VQA problem is still ambiguous and can be viewed from two perspectives: the technical perspective, measuring the perception of distortions; and the aesthetic perspective, which relates to preference and recommendation on contents. To understand how these two perspectives affect overall subjective opinions in UGC-VQA, we conduct a large-scale subjective study to collect human quality opinions on overall quality of videos as well as perceptions from aesthetic and technical perspectives. The collected Disentangled Video Quality Database (DIVIDE-3k) confirms that human quality opinions on UGC videos are universally and inevitably affected by both aesthetic and technical perspectives. In light of this, we propose the Disentangled Objective Video Quality Evaluator (DOVER) to learn the quality of UGC videos based on the two perspectives. The DOVER proves state-of-the-art performance in UGC-VQA under very high efficiency. With perspective opinions in DIVIDE-3k, we further propose DOVER++, the first approach to provide reliable clear-cut quality evaluations from a single aesthetic or technical perspective. Code at https://github.com/VQAssessment/DOVER.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/2.jpg" width="600">
  </p>

- **NIMA: Neural Image Assessment** [[paper](https://arxiv.org/pdf/1709.05424v2)] [[code](https://github.com/idealo/image-quality-assessment)]
  
  Automatically learned quality assessment for images has recently become a hot topic due to its usefulness in a wide variety of applications such as evaluating image capture pipelines, storage techniques and sharing media. Despite the subjective nature of this problem, most existing methods only predict the mean opinion score provided by datasets such as AVA [1] and TID2013 [2]. Our approach differs from others in that we predict the distribution of human opinion scores using a convolutional neural network. Our architecture also has the advantage of being significantly simpler than other methods with comparable performance. Our proposed approach relies on the success (and retraining) of proven, state-of-the-art deep object recognition networks. Our resulting network can be used to not only score images reliably and with high correlation to human perception, but also to assist with adaptation and optimization of photo editing/enhancement algorithms in a photographic pipeline. All this is done without need for a "golden" reference image, consequently allowing for single-image, semantic- and perceptually-aware, no-reference quality assessment.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/3.jpg" width="600">
  </p>

- **Composition-Preserving Deep Photo Aesthetics Assessment** [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Mai_Composition-Preserving_Deep_Photo_CVPR_2016_paper.pdf)] [[code](https://github.com/gautamMalu/Aesthetic_attributes_maps)]
  
  Photo aesthetics assessment is challenging. Deep convolutional neural network (ConvNet) methods have recently shown promising results for aesthetics assessment. The performance of these deep ConvNet methods, however, is often compromised by the constraint that the neural network only takes the fixed-size input. To accommodate this requirement, input images need to be transformed via cropping, scaling, or padding, which often damages image composition, reduces image resolution, or causes image distortion, thus compromising the aesthetics of the original images. In this paper, we present a composition-preserving deep ConvNet method that directly learns aesthetics features from the original input images without any image transformations. Specifically, our method adds an adaptive spatial pooling layer upon the regular convolution and pooling layers to directly handle input images with original sizes and aspect ratios. To allow for multi-scale feature extraction, we develop the Multi-Net Adaptive Spatial Pooling ConvNet architecture which consists of multiple sub-networks with different adaptive spatial pooling sizes and leverage a scene-based aggregation layer to effectively combine the predictions from multiple sub-networks. Our experiments on the large-scale aesthetics assessment benchmark (AVA) demonstrate that our method can significantly improve the state-of-the-art results in photo aesthetics assessment.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/4.jpg" width="600">
  </p>

- **Photo Aesthetics Ranking Network with Attributes and Content Adaptation** [[paper](https://arxiv.org/pdf/1606.01621v2)] [[code](https://github.com/aimerykong/deepImageAestheticsAnalysis)]
  
  Real-world applications could benefit from the ability to automatically generate a fine-grained ranking of photo aesthetics. However, previous methods for image aesthetics analysis have primarily focused on the coarse, binary categorization of images into high- or low-aesthetic categories. In this work, we propose to learn a deep convolutional neural network to rank photo aesthetics in which the relative ranking of photo aesthetics are directly modeled in the loss function. Our model incorporates joint learning of meaningful photographic attributes and image content information which can help regularize the complicated photo aesthetics rating problem. To train and analyze this model, we have assembled a new aesthetics and attributes database (AADB) which contains aesthetic scores and meaningful attributes assigned to each image by multiple human raters. Anonymized rater identities are recorded across images allowing us to exploit intra-rater consistency using a novel sampling strategy when computing the ranking loss of training image pairs. We show the proposed sampling strategy is very effective and robust in face of subjective judgement of image aesthetics by individuals with different aesthetic tastes. Experiments demonstrate that our unified model can generate aesthetic rankings that are more consistent with human ratings. To further validate our model, we show that by simply thresholding the estimated aesthetic scores, we are able to achieve state-or-the-art classification performance on the existing AVA dataset benchmark.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/5.jpg" width="600">
  </p>

- **ILGNet: Inception Modules with Connected Local and Global Features for Efficient Image Aesthetic Quality Classification using Domain Adaptation** [[paper](https://arxiv.org/pdf/1610.02256v3)] [[code](https://github.com/BestiVictory/ILGnet)]
  
  In this paper, we address a challenging problem of aesthetic image classification, which is to label an input image as high or low aesthetic quality. We take both the local and global features of images into consideration. A novel deep convolutional neural network named ILGNet is proposed, which combines both the Inception modules and an connected layer of both Local and Global features. The ILGnet is based on GoogLeNet. Thus, it is easy to use a pre-trained GoogLeNet for large-scale image classification problem and fine tune our connected layers on an large scale database of aesthetic related images: AVA, i.e. \emph{domain adaptation}. The experiments reveal that our model achieves the state of the arts in AVA database. Both the training and testing speeds of our model are higher than those of the original GoogLeNet.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/6.jpg" width="600">
  </p>

- **Effective Aesthetics Prediction with Multi-level Spatially Pooled Features** [[paper](https://arxiv.org/pdf/1904.01382v1)] [[code](https://github.com/subpic/ava-mlsp)]
  
  We propose an effective deep learning approach to aesthetics quality assessment that relies on a new type of pre-trained features, and apply it to the AVA data set, the currently largest aesthetics database. While previous approaches miss some of the information in the original images, due to taking small crops, down-scaling or warping the originals during training, we propose the first method that efficiently supports full resolution images as an input, and can be trained on variable input sizes. This allows us to significantly improve upon the state of the art, increasing the Spearman rank-order correlation coefficient (SRCC) of ground-truth mean opinion scores (MOS) from the existing best reported of 0.612 to 0.756. To achieve this performance, we extract multi-level spatially pooled (MLSP) features from all convolutional blocks of a pre-trained InceptionResNet-v2 network, and train a custom shallow Convolutional Neural Network (CNN) architecture on these new features.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/7.jpg" width="600">
  </p>

- **Personalized Image Aesthetics Assessment via Meta-Learning With Bilevel Gradient Optimization** [[paper](https://ieeexplore.ieee.org/abstract/document/9115059)] [[code](https://github.com/zhuhancheng/BLG-PIAA)]
  
  Typical image aesthetics assessment (IAA) is modeled for the generic aesthetics perceived by an average user. However, such generic aesthetics models neglect the fact that users aesthetic preferences vary significantly depending on their unique preferences. Therefore, it is essential to tackle the issue for personalized IAA (PIAA). Since PIAA is a typical small sample learning (SSL) problem, existing PIAA models are usually built by fine-tuning the well-established generic IAA (GIAA) models, which are regarded as prior knowledge. Nevertheless, this kind of prior knowledge based on average aesthetics fails to incarnate the aesthetic diversity of different people. In order to learn the shared prior knowledge when different people judge aesthetics, that is, learn how people judge image aesthetics, we propose a PIAA method based on meta-learning with bilevel gradient optimization (BLG-PIAA), which is trained using individual aesthetic data directly and generalizes to unknown users quickly. The proposed approach consists of two phases: 1) meta-training and 2) meta-testing. In meta-training, the aesthetics assessment of each user is regarded as a task, and the training set of each task is divided into two sets: 1) support set and 2) query set. Unlike traditional methods that train a GIAA model based on average aesthetics, we train an aesthetic meta-learner model by bilevel gradient updating from the support set to the query set using many users' PIAA tasks. In meta-testing, the aesthetic meta-learner model is fine-tuned using a small amount of aesthetic data of a target user to obtain the PIAA model. The experimental results show that the proposed method outperforms the state-of-the-art PIAA metrics, and the learned prior model of BLG-PIAA can be quickly adapted to unseen PIAA tasks.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/8.jpg" width="600">
  </p>

- **Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels** [[paper](https://arxiv.org/pdf/2312.17090v1)] [[code](https://github.com/q-future/q-align)]
  
  The explosion of visual content available online underscores the requirement for an accurate machine assessor to robustly evaluate scores across diverse types of visual contents. While recent studies have demonstrated the exceptional potentials of large multi-modality models (LMMs) on a wide range of related fields, in this work, we explore how to teach them for visual rating aligned with human opinions. Observing that human raters only learn and judge discrete text-defined levels in subjective studies, we propose to emulate this subjective process and teach LMMs with text-defined rating levels instead of scores. The proposed Q-Align achieves state-of-the-art performance on image quality assessment (IQA), image aesthetic assessment (IAA), as well as video quality assessment (VQA) tasks under the original LMM structure. With the syllabus, we further unify the three tasks into one model, termed the OneAlign. In our experiments, we demonstrate the advantage of the discrete-level-based syllabus over direct-score-based variants for LMMs. 
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/9.jpg" width="600">
  </p>



- **Image Composition Assessment with Saliency-augmented Multi-pattern Pooling** [[paper](https://arxiv.org/pdf/2104.03133v2)] [[code](https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB)]
  
  Image composition assessment is crucial in aesthetic assessment, which aims to assess the overall composition quality of a given image. However, to the best of our knowledge, there is neither dataset nor method specifically proposed for this task. In this paper, we contribute the first composition assessment dataset CADB with composition scores for each image provided by multiple professional raters. Besides, we propose a composition assessment network SAMP-Net with a novel Saliency-Augmented Multi-pattern Pooling (SAMP) module, which analyses visual layout from the perspectives of multiple composition patterns. We also leverage composition-relevant attributes to further boost the performance, and extend Earth Mover's Distance (EMD) loss to weighted EMD loss to eliminate the content bias. The experimental results show that our SAMP-Net can perform more favorably than previous aesthetic assessment approaches.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/10.jpg" width="600">
  </p>
  
### Datasets
- **GPD** [[paper](https://www.researchgate.net/publication/329329757_Gourmet_photography_dataset_for_aesthetic_assessment_of_food_images)] [[code](https://github.com/Openning07/GPA)]
  
  In this study, we present the Gourmet Photography Dataset (GPD),which is the rst large-scale dataset for aesthetic assessment offood photographs. We collect 12,000 food images together withhuman-annotated labels (i.e., aesthetically positive or negative) tobuild this dataset. We evaluate the performance of several popu-lar machine learning algorithms for aesthetic assessment of foodimages to verify the eectiveness and importance of our GPDdataset. Experimental results show that deep convolutional neuralnetworks trained on GPD can achieve comparable performancewith human experts in this task, even on unseen food photographs.Our experiments also provide insights to support further study andapplications related to visual analysis of food images
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/11.jpg" width="600">
  </p>

- **RPCD** [[paper](https://arxiv.org/pdf/2206.08614v3)] [[code](https://github.com/mediatechnologycenter/aestheval)]
  
  Computational inference of aesthetics is an ill-defined task due to its subjective nature. Many datasets have been proposed to tackle the problem by providing pairs of images and aesthetic scores based on human ratings. However, humans are better at expressing their opinion, taste, and emotions by means of language rather than summarizing them in a single number. In fact, photo critiques provide much richer information as they reveal how and why users rate the aesthetics of visual stimuli. In this regard, we propose the Reddit Photo Critique Dataset (RPCD), which contains tuples of image and photo critiques. RPCD consists of 74K images and 220K comments and is collected from a Reddit community used by hobbyists and professional photographers to improve their photography skills by leveraging constructive community feedback. The proposed dataset differs from previous aesthetics datasets mainly in three aspects, namely (i) the large scale of the dataset and the extension of the comments criticizing different aspects of the image, (ii) it contains mostly UltraHD images, and (iii) it can easily be extended to new data as it is collected through an automatic pipeline. To the best of our knowledge, in this work, we propose the first attempt to estimate the aesthetic quality of visual stimuli from the critiques. To this end, we exploit the polarity of the sentiment of criticism as an indicator of aesthetic judgment. We demonstrate how sentiment polarity correlates positively with the aesthetic judgment available for two aesthetic assessment benchmarks. Finally, we experiment with several models by using the sentiment scores as a target for ranking images.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/12.jpg" width="600">
  </p>

- **CADB** [[paper](https://arxiv.org/pdf/2104.03133v2)] [[code](https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB)]
  
  Image composition assessment is crucial in aesthetic assessment, which aims to assess the overall composition quality of a given image. However, to the best of our knowledge, there is neither dataset nor method specifically proposed for this task. In this paper, we contribute the first composition assessment dataset CADB with composition scores for each image provided by multiple professional raters. Besides, we propose a composition assessment network SAMP-Net with a novel Saliency-Augmented Multi-pattern Pooling (SAMP) module, which analyses visual layout from the perspectives of multiple composition patterns. We also leverage composition-relevant attributes to further boost the performance, and extend Earth Mover's Distance (EMD) loss to weighted EMD loss to eliminate the content bias. The experimental results show that our SAMP-Net can perform more favorably than previous aesthetic assessment approaches
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/13.jpg" width="600">
  </p>
  
## Artistic Image Generation
### Papers&Codes
- **$Z^*$ : Zero-shot Style Transfer via Attention Rearrangement** [[paper](https://arxiv.org/abs/2311.16491)] [[code](https://github.com/HolmesShuan/Zero-shot-Style-Transfer-via-Attention-Rearrangement)]
  
  Z-STAR is an innovative zero-shot (training-free) style transfer method that leverages the generative prior knowledge within a pre-trained diffusion model. By employing an attention rearrangement strategy, it effectively fuses content and style information without the need for retraining or tuning for each input style.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/14.jpg" width="600">
  </p>

- **A Unified Arbitrary Style Transfer Framework via Adaptive Contrastive Learning** [[paper](https://dl.acm.org/doi/pdf/10.1145/3605548)] [[code](https://github.com/zyxElsa/CAST_pytorch)]
  
  In this work, we tackle the challenging problem of arbitrary image style transfer using a novel style feature representation learning method. A suitable style representation, as a key component in image stylization tasks, is essential to achieve satisfactory results. Existing deep neural network based approaches achieve reasonable results with the guidance from second-order statistics such as Gram matrix of content features. However, they do not leverage sufficient style information, which results in artifacts such as local distortions and style inconsistency. To address these issues, we propose to learn style representation directly from image features instead of their second-order statistics, by analyzing the similarities and differences between multiple styles and considering the style distribution.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/15.jpg" width="600">
  </p>

- **ProSpect: Prompt Spectrum for Attribute-Aware Personalization of Diffusion Models** [[paper](https://arxiv.org/pdf/2305.16225)] [[code](https://github.com/zyxElsa/ProSpect)]
  
  Personalizing generative models offers a way to guide image generation with user-provided references. Current personalization methods can invert an object or concept into the textual conditioning space and compose new natural sentences for text-to-image diffusion models. However, representing and editing specific visual attributes like material, style, layout, etc. remains a challenge, leading to a lack of disentanglement and editability. To address this, we propose a novel approach that leverages the step-by-step generation process of diffusion models, which generate images from low- to high-frequency information, providing a new perspective on representing, generating, and editing images. We develop Prompt Spectrum Space P*, an expanded textual conditioning space, and a new image representation method called ProSpect. ProSpect represents an image as a collection of inverted textual token embeddings encoded from per-stage prompts, where each prompt corresponds to a specific generation stage (i.e., a group of consecutive steps) of the diffusion model. Experimental results demonstrate that P* and ProSpect offer stronger disentanglement and controllability compared to existing methods. We apply ProSpect in various personalized attribute-aware image generation applications, such as image/text-guided material/style/layout transfer/editing, achieving previously unattainable results with a single image input without fine-tuning the diffusion models.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/16.jpg" width="600">
  </p>

- **Inversion-Based Style Transfer with Diffusion Models** [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Inversion-Based_Style_Transfer_With_Diffusion_Models_CVPR_2023_paper.pdf)] [[code](https://github.com/zyxElsa/InST)]
  
  The artistic style within a painting is the means of expression, which includes not only the painting material, colors, and brushstrokes, but also the high-level attributes including semantic elements, object shapes, etc. Previous arbitrary example-guided artistic image generation methods often fail to control shape changes or convey elements. The pre-trained text-to-image synthesis diffusion probabilistic models have achieved remarkable quality, but it often requires extensive textual descriptions to accurately portray attributes of a particular painting. We believe that the uniqueness of an artwork lies precisely in the fact that it cannot be adequately explained with normal language.Our key idea is to learn artistic style directly from a single painting and then guide the synthesis without providing complex textual descriptions. Specifically, we assume style as a learnable textual description of a painting. We propose an inversion-based style transfer method (InST), which can efficiently and accurately learn the key information of an image, thus capturing and transferring the complete artistic style of a painting. We demonstrate the quality and efficiency of our method on numerous paintings of various artists and styles.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/17.jpg" width="600">
  </p>

- **Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning** [[paper](https://arxiv.org/abs/2205.09542)] [[code](https://github.com/zyxElsa/CAST_pytorch)]
  
  In this work, we tackle the challenging problem of arbitrary image style transfer using a novel style feature representation learning method. A suitable style representation, as a key component in image stylization tasks, is essential to achieve satisfactory results. Existing deep neural network based approaches achieve reasonable results with the guidance from second-order statistics such as Gram matrix of content features. However, they do not leverage sufficient style information, which results in artifacts such as local distortions and style inconsistency. To address these issues, we propose to learn style representation directly from image features instead of their second-order statistics, by analyzing the similarities and differences between multiple styles and considering the style distribution. Specifically, we present Contrastive Arbitrary Style Transfer (CAST), which is a new style representation learning and style transfer method via contrastive learning. Our framework consists of three key components, i.e., a multi-layer style projector for style code encoding, a domain enhancement module for effective learning of style distribution, and a generative network for image style transfer. We conduct qualitative and quantitative evaluations comprehensively to demonstrate that our approach achieves significantly better results compared to those obtained via state-of-the-art methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/18.jpg" width="600">
  </p>

- **$StyTr^2$ : Image Style Transfer with Transformers** [[paper](https://arxiv.org/abs/2105.14576)] [[code](https://github.com/diyiiyiii/StyTR-2)]
  
  The goal of image style transfer is to render an image with artistic features guided by a style reference while maintaining the original content. Owing to the locality in convolutional neural networks (CNNs), extracting and maintaining the global information of input images is difficult. Therefore, traditional neural style transfer methods face biased content representation. To address this critical issue, we take long-range dependencies of input images into account for image style transfer by proposing a transformer-based approach called StyTr2. In contrast with visual transformers for other vision tasks, StyTr2 contains two different transformer encoders to generate domain-specific sequences for content and style, respectively. Following the encoders, a multi-layer transformer decoder is adopted to stylize the content sequence according to the style sequence. We also analyze the deficiency of existing positional encoding methods and propose the content-aware positional encoding (CAPE), which is scale-invariant and more suitable for image style transfer tasks. Qualitative and quantitative experiments demonstrate the effectiveness of the proposed StyTr2 compared with state-of-the-art CNN-based and flow-based approaches.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/19.jpg" width="600">
  </p>

- **Arbitrary Style Transfer via Multi-Adaptation Network** [[paper](https://arxiv.org/abs/2005.13219)] [[code](https://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/)]
  
  Arbitrary style transfer is a significant topic with research value and application prospect. A desired style transfer, given a content image and referenced style painting, would render the content image with the color tone and vivid stroke patterns of the style painting while synchronously maintaining the detailed content structure information. Style transfer approaches would initially learn content and style representations of the content and style references and then generate the stylized images guided by these representations. In this paper, we propose the multi-adaptation network which involves two self-adaptation (SA) modules and one co-adaptation (CA) module: the SA modules adaptively disentangle the content and style representations, i.e., content SA module uses position-wise self-attention to enhance content representation and style SA module uses channel-wise self-attention to enhance style representation; the CA module rearranges the distribution of style representation based on content representation distribution by calculating the local similarity between the disentangled content and style features in a non-local fashion. Moreover, a new disentanglement loss function enables our network to extract main style patterns and exact content structures to adapt to various input images, respectively. Various qualitative and quantitative experiments demonstrate that the proposed multi-adaptation network leads to better results than the state-of-the-art style transfer methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/20.jpg" width="600">
  </p>

- **Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion** [[paper](https://arxiv.org/abs/2209.13360)] [[code](https://github.com/haha-lisa/MGAD-multimodal-guided-artwork-diffusion)]
  
  Digital art creation is getting more attention in the multimedia community for providing effective engagement of the public with art. Current digital art generation methods usually use single modality inputs as guidance, limiting the expressiveness of the model and the diversity of generated results. To solve this problem, we propose the multimodal guided artwork diffusion (MGAD) model, a diffusion-based digital artwork generation method that utilizes multimodal prompts as guidance to control the classifier-free diffusion model. Additionally, the contrastive language-image pretraining (CLIP) model is used to unify text and image modalities. However, the semantic content of multimodal prompts may conflict with each other, which leads to a collapse in generating progress. Extensive experimental results on the quality and quantity of the generated digital art paintings confirm the effectiveness of the combination of the diffusion model and multimodal guidance.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/21.jpg" width="600">
  </p>

- **Texture Networks: Feed-forward Synthesis of Textures and Stylized Images** [[paper](https://arxiv.org/pdf/1603.03417v1)] [[code](https://github.com/DmitryUlyanov/texture_nets)]
  
  Gatys et al. recently demonstrated that deep networks can generate beautiful textures and stylized images from a single texture example. However, their methods requires a slow and memory-consuming optimization process. We propose here an alternative approach that moves the computational burden to a learning stage. Given a single example of a texture, our approach trains compact feed-forward convolutional networks to generate multiple samples of the same texture of arbitrary size and to transfer artistic style from a given image to any other image. The resulting networks are remarkably light-weight and can generate textures of quality comparable to Gatys~et~al., but hundreds of times faster. More generally, our approach highlights the power and flexibility of generative feed-forward models trained with complex and expressive loss functions.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/22.jpg" width="600">
  </p>

- **A Neural Algorithm of Artistic Style** [[paper](https://arxiv.org/pdf/1508.06576v2)] [[code](https://github.com/jcjohnson/neural-style)]
  
  In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image. Thus far the algorithmic basis of this process is unknown and there exists no artificial system with similar capabilities. However, in other key areas of visual perception such as object and face recognition near-human performance was recently demonstrated by a class of biologically inspired vision models called Deep Neural Networks. Here we introduce an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images. Moreover, in light of the striking similarities between performance-optimised artificial neural networks and biological vision, our work offers a path forward to an algorithmic understanding of how humans create and perceive artistic imagery.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/23.jpg" width="600">
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
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/24.jpg" width="600">
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
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/25.jpg" width="600">
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
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/26.jpg" width="600">
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
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/27.jpg" width="600">
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
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/28.jpg" width="600">
  </p>

- **Instance Normalization: The Missing Ingredient for Fast Stylization** [[paper](https://arxiv.org/pdf/1607.08022v3)] [[code](https://github.com/DmitryUlyanov/texture_nets)]
  
  It this paper we revisit the fast stylization method introduced in Ulyanov et. al. (2016). We show how a small change in the stylization architecture results in a significant qualitative improvement in the generated images. The change is limited to swapping batch normalization with instance normalization, and to apply the latter both at training and testing times. The resulting method can be used to train high-performance architectures for real-time image generation.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/29.jpg" width="600">
  </p>

- **A Closed-form Solution to Photorealistic Image Stylization** [[paper](https://arxiv.org/pdf/1802.06474v5)] [[code](https://github.com/NVIDIA/FastPhotoStyle)]
  
  Photorealistic image stylization concerns transferring style of a reference photo to a content photo with the constraint that the stylized photo should remain photorealistic. While several photorealistic image stylization methods exist, they tend to generate spatially inconsistent stylizations with noticeable artifacts. In this paper, we propose a method to address these issues. The proposed method consists of a stylization step and a smoothing step. While the stylization step transfers the style of the reference photo to the content photo, the smoothing step ensures spatially consistent stylizations. Each of the steps has a closed-form solution and can be computed efficiently. We conduct extensive experimental validations. The results show that the proposed method generates photorealistic stylization outputs that are more preferred by human subjects as compared to those by the competing methods while running much faster.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/30.jpg" width="600">
  </p>

- **A Style-Aware Content Loss for Real-time HD Style Transfer** [[paper](https://arxiv.org/pdf/1807.10201v2)] [[code](https://github.com/CompVis/adaptive-style-transfer)]
  
  Recently, style transfer has received a lot of attention. While much of this research has aimed at speeding up processing, the approaches are still lacking from a principled, art historical standpoint: a style is more than just a single image or an artist, but previous work is limited to only a single instance of a style or shows no benefit from more images. Moreover, previous work has relied on a direct comparison of art in the domain of RGB images or on CNNs pre-trained on ImageNet, which requires millions of labeled object bounding boxes and can introduce an extra bias, since it has been assembled without artistic consideration. To circumvent these issues, we propose a style-aware content loss, which is trained jointly with a deep encoder-decoder network for real-time, high-resolution stylization of images and videos. We propose a quantitative measure for evaluating the quality of a stylized image and also have art historians rank patches from our approach against those from previous work. These and our qualitative results ranging from small image patches to megapixel stylistic images and videos show that our approach better captures the subtle nature in which a style affects content.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/31.jpg" width="600">
  </p>

- **Preserving Color in Neural Artistic Style Transfer** [[paper](https://arxiv.org/pdf/1606.05897v1)] [[code](https://github.com/cysmith/neural-style-tf)]
  
  This note presents an extension to the neural artistic style transfer algorithm (Gatys et al.). The original algorithm transforms an image to have the style of another given image. For example, a photograph can be transformed to have the style of a famous painting. Here we address a potential shortcoming of the original method: the algorithm transfers the colors of the original painting, which can alter the appearance of the scene in undesirable ways. We describe simple linear methods for transferring style while preserving colors.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/57.jpg" width="600">
  </p>

- **Multi-style Generative Network for Real-time Transfer** [[paper](https://arxiv.org/pdf/1703.06953v2)] [[code](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer)]
  
  Despite the rapid progress in style transfer, existing approaches using feed-forward generative network for multi-style or arbitrary-style transfer are usually compromised of image quality and model flexibility. We find it is fundamentally difficult to achieve comprehensive style modeling using 1-dimensional style embedding. Motivated by this, we introduce CoMatch Layer that learns to match the second order feature statistics with the target styles. With the CoMatch Layer, we build a Multi-style Generative Network (MSG-Net), which achieves real-time performance. We also employ an specific strategy of upsampled convolution which avoids checkerboard artifacts caused by fractionally-strided convolution. Our method has achieved superior image quality comparing to state-of-the-art approaches. The proposed MSG-Net as a general approach for real-time style transfer is compatible with most existing techniques including content-style interpolation, color-preserving, spatial control and brush stroke size control. MSG-Net is the first to achieve real-time brush-size control in a purely feed-forward manner for style transfer. Our implementations and pre-trained models for Torch, PyTorch and MXNet frameworks will be publicly available.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/58.jpg" width="600">
  </p>
### Datasets
- **FFHQ** [[paper](https://arxiv.org/pdf/1812.04948v3)] [[code](https://github.com/NVlabs/ffhq-dataset)]
  
  A dataset of human faces (Flickr-Faces-HQ, FFHQ) that offers much higher quality and covers considerably wider variation than existing high-resolution dataset
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/32.jpg" width="600">
  </p>
  
- **CelebA** [[paper](https://arxiv.org/pdf/1710.10196v3)] [[code](https://github.com/tkarras/progressive_growing_of_gans)]
  
  We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024^2. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/33.jpg" width="600">
  </p>

- **AFHQ** [[paper](https://arxiv.org/pdf/1912.01865v2)] [[code](https://github.com/clovaai/stargan-v2)]
  
  A good image-to-image translation model should learn a mapping between different visual domains while satisfying the following properties: 1) diversity of generated images and 2) scalability over multiple domains. Existing methods address either of the issues, having limited diversity or multiple models for all domains. We propose StarGAN v2, a single framework that tackles both and shows significantly improved results over the baselines. Experiments on CelebA-HQ and a new animal faces dataset (AFHQ) validate our superiority in terms of visual quality, diversity, and scalability. To better assess image-to-image translation models, we release AFHQ, high-quality animal faces with large inter- and intra-domain differences. The code, pretrained models, and dataset can be found at https://github.com/clovaai/stargan-v2.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/34.jpg" width="600">
  </p>


- **Fashion-MNIST** [[paper](https://arxiv.org/pdf/1708.07747v2)] [[code](https://github.com/zalandoresearch/fashion-mnist)]
  
  We present Fashion-MNIST, a new dataset comprising of 28x28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category. The training set has 60,000 images and the test set has 10,000 images. Fashion-MNIST is intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms, as it shares the same image size, data format and the structure of training and testing splits. The dataset is freely available at https://github.com/zalandoresearch/fashion-mnist
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/35.jpg" width="600">
  </p>
  
## Artistic Video Generation
### Papers&Codes
- **Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer** [[paper](https://arxiv.org/pdf/2305.05464)] [[code](https://github.com/haha-lisa/Style-A-Video)]
  
  This paper proposes a zero-shot video stylization method named Style-A-Video, which utilizes a generative pre-trained transformer with an image latent diffusion model to achieve a concise text-controlled video stylization.

  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/36.jpg" width="600">
  </p>

- **Arbitrary Video Style Transfer via Multi-Channel Correlation** [[paper](https://arxiv.org/abs/2009.08003)] [[code](https://github.com/diyiiyiii/MCCNet)]
  
  Video style transfer is getting more attention in AI community for its numerous applications such as augmented reality and animation productions. Compared with traditional image style transfer, performing this task on video presents new challenges: how to effectively generate satisfactory stylized results for any specified style, and maintain temporal coherence across frames at the same time. Towards this end, we propose Multi-Channel Correction network (MCCNet), which can be trained to fuse the exemplar style features and input content features for efficient style transfer while naturally maintaining the coherence of input videos. Specifically, MCCNet works directly on the feature space of style and content domain where it learns to rearrange and fuse style features based on their similarity with content features. The outputs generated by MCC are features containing the desired style patterns which can further be decoded into images with vivid style textures. Moreover, MCCNet is also designed to explicitly align the features to input which ensures the output maintains the content structures as well as the temporal continuity. To further improve the performance of MCCNet under complex light conditions, we also introduce the illumination loss during training. Qualitative and quantitative evaluations demonstrate that MCCNet performs well in both arbitrary video and image style transfer tasks.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/37.jpg" width="600">
  </p>

- **Everybody Dance Now** [[paper](https://arxiv.org/pdf/1808.07371v2)] [[code](https://github.com/carolineec/EverybodyDanceNow)]
  
  This paper presents a simple method for "do as I do" motion transfer: given a source video of a person dancing, we can transfer that performance to a novel (amateur) target after only a few minutes of the target subject performing standard moves. We approach this problem as video-to-video translation using pose as an intermediate representation. To transfer the motion, we extract poses from the source subject and apply the learned pose-to-appearance mapping to generate the target subject. We predict two consecutive frames for temporally coherent video results and introduce a separate pipeline for realistic face synthesis. Although our method is quite simple, it produces surprisingly compelling results (see video). This motivates us to also provide a forensics tool for reliable synthetic content detection, which is able to distinguish videos synthesized by our system from real data. In addition, we release a first-of-its-kind open-source dataset of videos that can be legally used for training and motion transfer.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/38.jpg" width="600">
  </p>

- **Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation** [[paper](https://arxiv.org/pdf/1811.09393v4)] [[code](https://github.com/thunil/TecoGAN)]
  
  Our work explores temporal self-supervision for GAN-based video generation tasks. While adversarial training successfully yields generative models for a variety of areas, temporal relationships in the generated data are much less explored. Natural temporal changes are crucial for sequential generation tasks, e.g. video super-resolution and unpaired video translation. For the former, state-of-the-art methods often favor simpler norm losses such as 
 over adversarial training. However, their averaging nature easily leads to temporally smooth results with an undesirable lack of spatial detail. For unpaired video translation, existing approaches modify the generator networks to form spatio-temporal cycle consistencies. In contrast, we focus on improving learning objectives and propose a temporally self-supervised algorithm. For both tasks, we show that temporal adversarial learning is key to achieving temporally coherent solutions without sacrificing spatial detail. We also propose a novel Ping-Pong loss to improve the long-term temporal consistency. It effectively prevents recurrent networks from accumulating artifacts temporally without depressing detailed features. Additionally, we propose a first set of metrics to quantitatively evaluate the accuracy as well as the perceptual quality of the temporal evolution. A series of user studies confirm the rankings computed with these metrics. Code, data, models, and results are provided at https://github.com/thunil/TecoGAN. The project page https://ge.in.tum.de/publications/2019-tecogan-chu/ contains supplemental materials.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/39.jpg" width="600">
  </p>

- **CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer** [[paper](https://arxiv.org/pdf/2207.04808v4)] [[code](https://github.com/JarrentWu1031/CCPL)]
  
  In this paper, we aim to devise a universally versatile style transfer method capable of performing artistic, photo-realistic, and video style transfer jointly, without seeing videos during training. Previous single-frame methods assume a strong constraint on the whole image to maintain temporal consistency, which could be violated in many cases. Instead, we make a mild and reasonable assumption that global inconsistency is dominated by local inconsistencies and devise a generic Contrastive Coherence Preserving Loss (CCPL) applied to local patches. CCPL can preserve the coherence of the content source during style transfer without degrading stylization. Moreover, it owns a neighbor-regulating mechanism, resulting in a vast reduction of local distortions and considerable visual quality improvement. Aside from its superior performance on versatile style transfer, it can be easily extended to other tasks, such as image-to-image translation. Besides, to better fuse content and style features, we propose Simple Covariance Transformation (SCT) to effectively align second-order statistics of the content feature with the style feature. Experiments demonstrate the effectiveness of the resulting model for versatile style transfer, when armed with CCPL.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/40.jpg" width="600">
  </p>

- **Artistic style transfer for videos and spherical images** [[paper](https://arxiv.org/pdf/1708.04538v3)] [[code](https://github.com/manuelruder/fast-artistic-videos)]
  
  Manually re-drawing an image in a certain artistic style takes a professional artist a long time. Doing this for a video sequence single-handedly is beyond imagination. We present two computational approaches that transfer the style from one image (for example, a painting) to a whole video sequence. In our first approach, we adapt to videos the original image style transfer technique by Gatys et al. based on energy minimization. We introduce new ways of initialization and new loss functions to generate consistent and stable stylized video sequences even in cases with large motion and strong occlusion. Our second approach formulates video stylization as a learning problem. We propose a deep network architecture and training procedures that allow us to stylize arbitrary-length videos in a consistent and stable way, and nearly in real time. We show that the proposed methods clearly outperform simpler baselines both qualitatively and quantitatively. Finally, we propose a way to adapt these approaches also to 360 degree images and videos as they emerge with recent virtual reality hardware.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/41.jpg" width="600">
  </p>

- **Music2Video: Automatic Generation of Music Video with fusion of audio and text** [[paper](https://arxiv.org/pdf/2201.03809v2)] [[code](https://github.com/joeljang/music2video)]
  
  Creation of images using generative adversarial networks has been widely adapted into multi-modal regime with the advent of multi-modal representation models pre-trained on large corpus. Various modalities sharing a common representation space could be utilized to guide the generative models to create images from text or even from audio source. Departing from the previous methods that solely rely on either text or audio, we exploit the expressiveness of both modality. Based on the fusion of text and audio, we create video whose content is consistent with the distinct modalities that are provided. A simple approach to automatically segment the video into variable length intervals and maintain time consistency in generated video is part of our method. Our proposed framework for generating music video shows promising results in application level where users can interactively feed in music source and text source to create artistic music videos. Our code is available at https://github.com/joeljang/music2video.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/42.jpg" width="600">
  </p>

- **Video Diffusion Models** [[paper](https://arxiv.org/pdf/2204.03458v2)] [[code](https://github.com/lucidrains/video-diffusion-pytorch)]
  
  Generating temporally coherent high fidelity video is an important milestone in generative modeling research. We make progress towards this milestone by proposing a diffusion model for video generation that shows very promising initial results. Our model is a natural extension of the standard image diffusion architecture, and it enables jointly training from image and video data, which we find to reduce the variance of minibatch gradients and speed up optimization. To generate long and higher resolution videos we introduce a new conditional sampling technique for spatial and temporal video extension that performs better than previously proposed methods. We present the first results on a large text-conditioned video generation task, as well as state-of-the-art results on established benchmarks for video prediction and unconditional video generation.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/43.jpg" width="600">
  </p>



- **Consistent Video Style Transfer via Relaxation and Regularization** [[paper](https://ieeexplore.ieee.org/document/9204808)] [[code](https://github.com/daooshee/ReReVST-Code)]
  
  In recent years, neural style transfer has attracted more and more attention, especially for image style transfer. However, temporally consistent style transfer for videos is still a challenging problem. Existing methods, either relying on a significant amount of video data with optical flows or using single-frame regularizers, fail to handle strong motions or complex variations, therefore have limited performance on real videos. In this article, we address the problem by jointly considering the intrinsic properties of stylization and temporal consistency. We first identify the cause of the conflict between style transfer and temporal consistency, and propose to reconcile this contradiction by relaxing the objective function, so as to make the stylization loss term more robust to motions. Through relaxation, style transfer is more robust to inter-frame variation without degrading the subjective effect. Then, we provide a novel formulation and understanding of temporal consistency. Based on the formulation, we analyze the drawbacks of existing training strategies and derive a new regularization. We show by experiments that the proposed regularization can better balance the spatial and temporal performance. Based on relaxation and regularization, we design a zero-shot video style transfer framework. Moreover, for better feature migration, we introduce a new module to dynamically adjust inter-channel distributions. Quantitative and qualitative results demonstrate the superiority of our method over state-of-the-art style transfer methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/44.jpg" width="600">
  </p>

- **DwNet: Dense warp-based network for pose-guided human video generation** [[paper](https://arxiv.org/pdf/1910.09139v1)] [[code](https://github.com/ubc-vision/DwNet)]
  
  Generation of realistic high-resolution videos of human subjects is a challenging and important task in computer vision. In this paper, we focus on human motion transfer - generation of a video depicting a particular subject, observed in a single image, performing a series of motions exemplified by an auxiliary (driving) video. Our GAN-based architecture, DwNet, leverages dense intermediate pose-guided representation and refinement process to warp the required subject appearance, in the form of the texture, from a source image into a desired pose. Temporal consistency is maintained by further conditioning the decoding process within a GAN on the previously generated frame. In this way a video is generated in an iterative and recurrent fashion. We illustrate the efficacy of our approach by showing state-of-the-art quantitative and qualitative performance on two benchmark datasets: TaiChi and Fashion Modeling. The latter is collected by us and will be made publicly available to the community.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/45.jpg" width="600">
  </p>

- **ReCoNet: Real-time Coherent Video Style Transfer Network** [[paper](https://arxiv.org/pdf/1807.01197v2)] [[code](https://github.com/safwankdb/ReCoNet-PyTorch)]
  
  Image style transfer models based on convolutional neural networks usually suffer from high temporal inconsistency when applied to videos. Some video style transfer models have been proposed to improve temporal consistency, yet they fail to guarantee fast processing speed, nice perceptual style quality and high temporal consistency at the same time. In this paper, we propose a novel real-time video style transfer model, ReCoNet, which can generate temporally coherent style transfer videos while maintaining favorable perceptual styles. A novel luminance warping constraint is added to the temporal loss at the output level to capture luminance changes between consecutive frames and increase stylization stability under illumination effects. We also propose a novel feature-map-level temporal loss to further enhance temporal consistency on traceable objects. Experimental results indicate that our model exhibits outstanding performance both qualitatively and quantitatively.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/46.jpg" width="600">
  </p>

- **FateZero: Fusing Attentions for Zero-shot Text-based Video Editing** [[paper](https://arxiv.org/pdf/2303.09535v3)] [[code](https://github.com/chenyangqiqi/fatezero)]
  
  The diffusion-based generative models have achieved remarkable success in text-based image generation. However, since it contains enormous randomness in generation progress, it is still challenging to apply such models for real-world visual content editing, especially in videos. In this paper, we propose FateZero, a zero-shot text-based editing method on real-world videos without per-prompt training or use-specific mask. To edit videos consistently, we propose several techniques based on the pre-trained models. Firstly, in contrast to the straightforward DDIM inversion technique, our approach captures intermediate attention maps during inversion, which effectively retain both structural and motion information. These maps are directly fused in the editing process rather than generated during denoising. To further minimize semantic leakage of the source video, we then fuse self-attentions with a blending mask obtained by cross-attention features from the source prompt. Furthermore, we have implemented a reform of the self-attention mechanism in denoising UNet by introducing spatial-temporal attention to ensure frame consistency. Yet succinct, our method is the first one to show the ability of zero-shot text-driven video style and local attribute editing from the trained text-to-image model. We also have a better zero-shot shape-aware editing ability based on the text-to-video model. Extensive experiments demonstrate our superior temporal consistency and editing capability than previous works.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/47.jpg" width="600">
  </p>

- **VToonify: Controllable High-Resolution Portrait Video Style Transfer** [[paper](https://arxiv.org/pdf/2209.11224v3)] [[code](https://github.com/williamyang1991/vtoonify)]
  
  Generating high-quality artistic portrait videos is an important and desirable task in computer graphics and vision. Although a series of successful portrait image toonification models built upon the powerful StyleGAN have been proposed, these image-oriented methods have obvious limitations when applied to videos, such as the fixed frame size, the requirement of face alignment, missing non-facial details and temporal inconsistency. In this work, we investigate the challenging controllable high-resolution portrait video style transfer by introducing a novel VToonify framework. Specifically, VToonify leverages the mid- and high-resolution layers of StyleGAN to render high-quality artistic portraits based on the multi-scale content features extracted by an encoder to better preserve the frame details. The resulting fully convolutional architecture accepts non-aligned faces in videos of variable size as input, contributing to complete face regions with natural motions in the output. Our framework is compatible with existing StyleGAN-based image toonification models to extend them to video toonification, and inherits appealing features of these models for flexible style control on color and intensity. This work presents two instantiations of VToonify built upon Toonify and DualStyleGAN for collection-based and exemplar-based portrait video style transfer, respectively. Extensive experimental results demonstrate the effectiveness of our proposed VToonify framework over existing methods in generating high-quality and temporally-coherent artistic portrait videos with flexible style controls.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/48.jpg" width="600">
  </p>

- **Two Birds, One Stone: A Unified Framework for Joint Learning of Image and Video Style Transfers** [[paper](https://arxiv.org/pdf/2304.11335v1)] [[code](https://github.com/NevSNev/UniST)]
  
  Current arbitrary style transfer models are limited to either image or video domains. In order to achieve satisfying image and video style transfers, two different models are inevitably required with separate training processes on image and video domains, respectively. In this paper, we show that this can be precluded by introducing UniST, a Unified Style Transfer framework for both images and videos. At the core of UniST is a domain interaction transformer (DIT), which first explores context information within the specific domain and then interacts contextualized domain information for joint learning. In particular, DIT enables exploration of temporal information from videos for the image style transfer task and meanwhile allows rich appearance texture from images for video style transfer, thus leading to mutual benefits. Considering heavy computation of traditional multi-head self-attention, we present a simple yet effective axial multi-head self-attention (AMSA) for DIT, which improves computational efficiency while maintains style transfer performance. To verify the effectiveness of UniST, we conduct extensive experiments on both image and video style transfer tasks and show that UniST performs favorably against state-of-the-art approaches on both tasks. Our code and results will be released.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/49.jpg" width="600">
  </p>

- **Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer** [[paper](https://arxiv.org/pdf/2305.05464v1)] [[code](https://github.com/haha-lisa/style-a-video)]
  
  Large-scale text-to-video diffusion models have demonstrated an exceptional ability to synthesize diverse videos. However, due to the lack of extensive text-to-video datasets and the necessary computational resources for training, directly applying these models for video stylization remains difficult. Also, given that the noise addition process on the input content is random and destructive, fulfilling the style transfer task's content preservation criteria is challenging. This paper proposes a zero-shot video stylization method named Style-A-Video, which utilizes a generative pre-trained transformer with an image latent diffusion model to achieve a concise text-controlled video stylization. We improve the guidance condition in the denoising process, establishing a balance between artistic expression and structure preservation. Furthermore, to decrease inter-frame flicker and avoid the formation of additional artifacts, we employ a sampling optimization and a temporal consistency module. Extensive experiments show that we can attain superior content preservation and stylistic performance while incurring less consumption than previous solutions. 
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/50.jpg" width="600">
  </p>

- **Control-A-Video: Controllable Text-to-Video Diffusion Models with Motion Prior and Reward Feedback Learning** [[paper](https://arxiv.org/pdf/2305.13840v3)] [[code](https://github.com/weifeng-chen/control-a-video)]
  
  Recent advances in text-to-image (T2I) diffusion models have enabled impressive image generation capabilities guided by text prompts. However, extending these techniques to video generation remains challenging, with existing text-to-video (T2V) methods often struggling to produce high-quality and motion-consistent videos. In this work, we introduce Control-A-Video, a controllable T2V diffusion model that can generate videos conditioned on text prompts and reference control maps like edge and depth maps. To tackle video quality and motion consistency issues, we propose novel strategies to incorporate content prior and motion prior into the diffusion-based generation process. Specifically, we employ a first-frame condition scheme to transfer video generation from the image domain. Additionally, we introduce residual-based and optical flow-based noise initialization to infuse motion priors from reference videos, promoting relevance among frame latents for reduced flickering. Furthermore, we present a Spatio-Temporal Reward Feedback Learning (ST-ReFL) algorithm that optimizes the video diffusion model using multiple reward models for video quality and motion consistency, leading to superior outputs. Comprehensive experiments demonstrate that our framework generates higher-quality, more consistent videos compared to existing state-of-the-art methods in controllable text-to-video generation
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/51.jpg" width="600">
  </p>
  
- **CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer** [[paper](https://arxiv.org/pdf/2303.17867v1)] [[code](https://github.com/linfengWen98/CAP-VSTNet)]
  
  Content affinity loss including feature and pixel affinity is a main problem which leads to artifacts in photorealistic and video style transfer. This paper proposes a new framework named CAP-VSTNet, which consists of a new reversible residual network and an unbiased linear transform module, for versatile style transfer. This reversible residual network can not only preserve content affinity but not introduce redundant information as traditional reversible networks, and hence facilitate better stylization. Empowered by Matting Laplacian training loss which can address the pixel affinity loss problem led by the linear transform, the proposed framework is applicable and effective on versatile style transfer. Extensive experiments show that CAP-VSTNet can produce better qualitative and quantitative results in comparison with the state-of-the-art methods.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/59.jpg" width="600">
  </p>
  - **WAIT: Feature Warping for Animation to Illustration video Translation using GANs** [[paper](https://arxiv.org/pdf/2310.04901v1)] [[code](https://github.com/giddyyupp/wait)]
  
  In this paper, we explore a new domain for video-to-video translation. Motivated by the availability of animation movies that are adopted from illustrated books for children, we aim to stylize these videos with the style of the original illustrations. Current state-of-the-art video-to-video translation models rely on having a video sequence or a single style image to stylize an input video. We introduce a new problem for video stylizing where an unordered set of images are used. This is a challenging task for two reasons: i) we do not have the advantage of temporal consistency as in video sequences; ii) it is more difficult to obtain consistent styles for video frames from a set of unordered images compared to using a single image. Most of the video-to-video translation methods are built on an image-to-image translation model, and integrate additional networks such as optical flow, or temporal predictors to capture temporal relations. These additional networks make the model training and inference complicated and slow down the process. To ensure temporal coherency in video-to-video style transfer, we propose a new generator network with feature warping layers which overcomes the limitations of the previous methods. We show the effectiveness of our method on three datasets both qualitatively and quantitatively. 
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/60.jpg" width="600">
  </p>
  
### Datasets
- **ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation** [[paper](https://arxiv.org/pdf/2406.18522v2)] [[code](https://github.com/pku-yuangroup/chronomagic-bench)]
  
  We propose a novel text-to-video (T2V) generation benchmark, ChronoMagic-Bench, to evaluate the temporal and metamorphic capabilities of the T2V models (e.g. Sora and Lumiere) in time-lapse video generation. In contrast to existing benchmarks that focus on visual quality and textual relevance of generated videos, ChronoMagic-Bench focuses on the model's ability to generate time-lapse videos with significant metamorphic amplitude and temporal coherence. The benchmark probes T2V models for their physics, biology, and chemistry capabilities, in a free-form text query. For these purposes, ChronoMagic-Bench introduces 1,649 prompts and real-world videos as references, categorized into four major types of time-lapse videos: biological, human-created, meteorological, and physical phenomena, which are further divided into 75 subcategories. This categorization comprehensively evaluates the model's capacity to handle diverse and complex transformations. To accurately align human preference with the benchmark, we introduce two new automatic metrics, MTScore and CHScore, to evaluate the videos' metamorphic attributes and temporal coherence. MTScore measures the metamorphic amplitude, reflecting the degree of change over time, while CHScore assesses the temporal coherence, ensuring the generated videos maintain logical progression and continuity. Based on ChronoMagic-Bench, we conduct comprehensive manual evaluations of ten representative T2V models, revealing their strengths and weaknesses across different categories of prompts, and providing a thorough evaluation framework that addresses current gaps in video generation research. Moreover, we create a large-scale ChronoMagic-Pro dataset, containing 460k high-quality pairs of 720p time-lapse videos and detailed captions ensuring high physical pertinence and large metamorphic amplitude. [Homepage](https://pku-yuangroup.github.io/ChronoMagic-Bench/).
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/52.jpg" width="600">
  </p>


- **UCF101** [[paper](https://arxiv.org/pdf/1212.0402v1)] [[code](https://www.crcv.ucf.edu/data/UCF101.php)]
  
  We introduce UCF101 which is currently the largest dataset of human actions. It consists of 101 action classes, over 13k clips and 27 hours of video data. The database consists of realistic user uploaded videos containing camera motion and cluttered background. Additionally, we provide baseline action recognition results on this new dataset using standard bag of words approach with overall performance of 44.5%. To the best of our knowledge, UCF101 is currently the most challenging dataset of actions due to its large number of classes, large number of clips and also unconstrained nature of such clips.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/53.jpg" width="600">
  </p>

- **LAION-400M** [[paper](https://arxiv.org/pdf/2111.02114v1)] [[code](https://github.com/mlfoundations/open_clip)]
  
  Multi-modal language-vision models trained on hundreds of millions of image-text pairs (e.g. CLIP, DALL-E) gained a recent surge, showing remarkable capability to perform zero- or few-shot learning and transfer even in absence of per-sample labels on target image data. Despite this trend, to date there has been no publicly available datasets of sufficient scale for training such models from scratch. To address this issue, in a community effort we build and release for public LAION-400M, a dataset with CLIP-filtered 400 million image-text pairs, their CLIP embeddings and kNN indices that allow efficient similarity search.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/54.jpg" width="600">
  </p>

- **Kinetics** [[paper](https://arxiv.org/pdf/1705.06950v1)] [[code](https://github.com/google-deepmind/kinetics-i3d/tree/master)]
  
  We describe the DeepMind Kinetics human action video dataset. The dataset contains 400 human action classes, with at least 400 video clips for each action. Each clip lasts around 10s and is taken from a different YouTube video. The actions are human focussed and cover a broad range of classes including human-object interactions such as playing instruments, as well as human-human interactions such as shaking hands. We describe the statistics of the dataset, how it was collected, and give some baseline performance figures for neural network architectures trained and tested for human action classification on this dataset. We also carry out a preliminary analysis of whether imbalance in the dataset leads to bias in the classifiers.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/55.jpg" width="600">
  </p>

- **How2Sign** [[paper](https://arxiv.org/pdf/2008.08143v2)] [[code](https://github.com/how2sign/how2sign.github.io)]
  
  One of the factors that have hindered progress in the areas of sign language recognition, translation, and production is the absence of large annotated datasets. Towards this end, we introduce How2Sign, a multimodal and multiview continuous American Sign Language (ASL) dataset, consisting of a parallel corpus of more than 80 hours of sign language videos and a set of corresponding modalities including speech, English transcripts, and depth. A three-hour subset was further recorded in the Panoptic studio enabling detailed 3D pose estimation. To evaluate the potential of How2Sign for real-world impact, we conduct a study with ASL signers and show that synthesized videos using our dataset can indeed be understood. The study further gives insights on challenges that computer vision should address in order to make progress in this field.
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/blob/main/images/56.jpg" width="600">
  </p>







