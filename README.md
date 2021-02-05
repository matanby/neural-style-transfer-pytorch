# Neural Style-Transfer PyTorch

This is a simple and minimalistic PyTorch implementation of 
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et. al (2015).

The paper presents a method for combining the content from one image, with the style of another image.
This method uses a pretrained convolutional neural-network (VGG19) for extracting multi-scale features from both images.
These features are then combined to create a stylized version of an input content image.

<div align="center">
 <img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/2.jpg" height="250px">
 <img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/5_cropped.jpg" height="250px">
 <br>
 <img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/2_5.jpg" width="395px">
</div>

---

### Prerequisites:
* Python 3
* CUDA + CUDNN (for GPU acceleration)

### Installation:
1. Clone this repository:
```
git clone https://github.com/matanby/neural-style-transfer-pytorch
```

2. Install PIP requirements:
```shell script
python3 -m virtualenv .venv
source .venv/bin/activate 
pip install -r neural-style-transfer-pytorch/requirements.py
```

---

### Usage:
#### From command-line:
```shell script
python run.py [PATH_TO_CONTENT_IMAGE] [PATH_TO_STYLE_IMAGE] [PATH_TO_STYLIZED_OUTPUT]
```

It is also possible to override the algorithm's default hyper-parameters values,
 by providing additional CLI arguments, for example:
 
```shell script
python run.py images/content/1.jpg images/style/1.jpg images/stylized/1.jpg --iterations 1000 --lambda_style 200
```

Complete list of hyper-parameters:    
* `lambda_content`: the weight of the content term in the total loss. default = 1.
* `lambda_style`: the weight of the style term in the total loss. 
   empirically good range: 10 - 100,000. default = 100.
* `lambda_tv`: the weight of the generated image's total variation 
   in the total loss. empirically good range: 0 - 1,000. default = 10.
* `step_size`: the size of each step of the optimization process. default = 0.1.
* `iterations`: number of optimization iterations. default = 500.
* `content_block_weights`: the weight of each convolutional block in the content loss.
   These five numbers refer to the following five activations of
   the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
   default = (0.0, 0.0, 0.0, 1.0, 0.0).
* `style_block_weights`: the weight of each convolutional block in the style loss.
   These five numbers refer to the following five activations of
   the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
   default = (1/5, 1/5, 1/5, 1/5, 1/5).
* `random_initial_image`: whether or not the optimization process should start with
    a random initial image (True), or the input content image (False). default = false.
* `max_input_dim`: the maximal allowed input image dimension. input images of
   which max(H,W) is larger than this number will be downscaled appropriately.
   this also defines the dimension of the generated stylized image.
   Raising this value will allow creating larger stylized images, but
   will also require more time and memory. default = 512.
* `save_interval`: the interval (number of iterations) after which an intermediate
   result of the stylized image will be saved to the disk. default = 50.
   
---

#### Programmatically:

Use the `Stylizer` class to create stylized images programmatically. For example:
```python
import image_utils
from stylizer import Stylizer, StylizerConfig

stylizer = Stylizer()
content = image_utils.load('images/content/7.jpg')
style = image_utils.load('images/style/5.jpg')

# Create multiple stylized versions of the content
# image with different values for `lambda_style`
for lambda_style in (100, 1_000, 10_000):
    config = StylizerConfig(lambda_style=lambda_style)
    stylized = stylizer.stylize(content, style, config)
    image_utils.save(stylized, f'images/stylized/7_5_lambda_style_{lambda_style}.jpg')
```

The code above generates the following images:
<div align="center">
<img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/7_5_lambda_style_100.jpg" height="400px">
<img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/7_5_lambda_style_1000.jpg" height="400px">
<img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/7_5_lambda_style_10000.jpg" height="400px">
</div>



---

### Examples:
<div align="center">
<table>
<tr>

<td vlign="center"><b>Content Image</b></td>
<td vlign="center"><b>Style Image</b></td>
<td vlign="center"><b>Stylized Result</b></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/1.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/2.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/1_2.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/2.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/5.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/2_5.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/3.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/10.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/3_10.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/4.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/3.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/4_3.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/5.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/4.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/5_4.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/6.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/11.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/6_11.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/7.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/1.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/7_1.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/content/8.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/style/6.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/neural-style-transfer-pytorch/raw/master/images/stylized/8_6.jpg" alt="content" width="200"/></td>
</tr>
</table>
</div>
