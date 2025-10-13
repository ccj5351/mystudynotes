---
layout: post
title:  "Quantization Fundamentals Study - 01"
date:   2025-10-12 12:50:12 -0400
categories: quantization, on-device AI
---

This is the study notes of the online course [Quantization Fundamentals with Hugging Face](https://learn.deeplearning.ai/courses/quantization-fundamentals/lesson/psgkw/introduction)

## Data Types and Sizes - Integer

<div align="center">
<img  src="{{ site.baseurl }}{% link docs/quantization/imgs/integer-def.png %}" alt="integer"  width="500"  />
<br><figcaption>
Fig. 1. Integer.
</figcaption>
</div>

To check the integer ranges information in PyTorch, we can use "torch.iinfo".

```python
# Information of `8-bit unsigned integer`
torch.iinfo(torch.uint8)
>>> iinfo(min=0, max=255, dtype=uint8)
# Information of `8-bit (signed) integer`
torch.iinfo(torch.int8)
>>> iinfo(min=-128, max=127, dtype=int8)
# Information of `16-bit (signed) integer`
torch.iinfo(torch.int16)
>>> iinfo(min=-32768, max=32767, dtype=int16)
```

## Data Types and Sizes - Floating Point
Three components in floating point:
- Sign (1 bit): The leftmost bit. `0` for positive, `1` for negative.
- Exponent, or range: impact the representable range of the number
- Mantissa or Fraction (precision): impact on the precision of the number

FP32, BF16, FP16, FP8 are floating point format with a specific number of bits for exponent and the fraction.

### FP32

<div align="center">
<img  src="{{ site.baseurl }}{% link docs/quantization/imgs/fp32-def.png %}" alt="FP32"  width="500"  />
<br><figcaption>
Fig. 2. Floating point 32-bit.
</figcaption>
</div>

### FP16

<div align="center">
<img  src="{{ site.baseurl }}{% link docs/quantization/imgs/fp16-def.png %}" alt="FP16"  width="500"  />
<br><figcaption>
Fig. 3. Floating point 16-bit.
</figcaption>
</div>


### BF16

<div align="center">
<img  src="{{ site.baseurl }}{% link docs/quantization/imgs/bf16-def.png %}" alt="BF16"  width="500"  />
<br><figcaption>
Fig. 4. Brain Floating point 16-bit.
</figcaption>
</div>

### TF32 (TensorFloat-32)
[TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) in the A100 GPU Accelerates AI Training. NVIDIA's Ampere architecture with TF32 speeds single-precision work, maintaining accuracy and using no new code.

<div align="center">
<img  src="{{ site.baseurl }}{% link docs/quantization/imgs/tf32-Mantissa-chart-hi-res-FINAL.png %}" alt="TF32"  width="500"  />
<br><figcaption>
Fig. 5. TensorFloat-32, TF32, strikes a balance that delivers performance with range and accuracy.
</figcaption>
</div>

### Comparison

| Data Type | Precision | maximum |
|----------|----------|----------|
|  FP32 |  best |  ~$10^{+38}$ |
|  FP16 |  better |  ~$10^{04}$ |
|  BF16 |  good |  ~$10^{38}$ 🤗 |


### Coding Example: 

See the [notebook](https://github.com/ccj5351/mystudynotes/blob/master/docs/quantization/quant_lesson01_data_types.ipynb) for how to decode decimal float number to binary string, and vice versa. 

### Floating Point in PyTorch



To check the float ranges information in PyTorch, we can use "torch.iinfo".

| Data Type | torch.dtype | torch.dtype alias |
|----------|----------|----------|
|  16-bit floating point  |  torch.float16 |  torch.half |
|  16-bit brain floating point  |  torch.bfloat16 |  n/a |
|  32-bit floating point  |  torch.float32 |  torch.float |
|  64-bit floating point  |  torch.float64 |  torch.double |
