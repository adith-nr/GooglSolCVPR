import requests
import torch
from PIL import Image

from isc_feature_extractor import create_model

recommended_weight_name = 'isc_ft_v107'
model, preprocessor = create_model(weight_name=recommended_weight_name, device='cpu')

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
x = preprocessor(image).unsqueeze(0)

y = model(x)
print(y.shape)  # => torch.Size([1, 256])
