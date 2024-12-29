from urllib.request import urlopen

import timm
from PIL import Image

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)

model = timm.create_model(
    "vit_large_patch16_224.augreg_in21k",
    pretrained=True,
    num_classes=0,
)
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))


output = model.forward_head(output, pre_logits=True)

print(output)
print(output.shape)
