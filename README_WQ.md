快速开始

```bash
conda activate grad_cam
python usage_examples/clip_example.py
```

## usage_examples/qwen2_5vl_encoder_example.py

原理：参考<https://github.com/bytedance/LVLM_Interpretation>获取生成式VLM的关键token，并计算这些关键token对于某一层图像隐层特征的显著性分数

如输入文本："英文字母的颜色"。模型生成："The English letters in the image are black."。获取关键token："black"。计算关键token对图片patch隐层特征（ViT merge前或者merge后）的显著性分数

## usage_examples/qwen2_5vl_decoder_example.py

原理与encoder版一致，只是目标层选的是decoder

## usage_examples/siglip2_example.py

## usage_examples/metaclip2_example.py

## usage_examples/lseg_example.py

## usage_examples/clip_surgery_example.py

## usage_examples/clip_example.py

## usage_examples/blip_omnixai_gradcam_example.py

## notebooks/omdet_eigencam.ipynb

## notebooks/groundingdino_gradcam.ipynb