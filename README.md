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

### Datasets

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






13. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

14. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

15. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

16. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

17. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

18. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

19. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

20. **** [[paper]()] [[code]()]
  
  Z-STAR
  
  <p align="center">
    <img src="https://github.com/joeysbase/AI-for-HASS/edit/main/images/.png" width="600">
  </p>

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

