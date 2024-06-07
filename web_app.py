from PIL import Image
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st

class SupportSet():
    def __init__(self, support_dir, class_name, label, cro_name, description, transform=None):
        self.class_name = class_name
        self.label = label
        self.cro_name = cro_name
        self.description = description
        self.images = []

        images_paths = [os.path.join(support_dir, filename) for filename in os.listdir(support_dir)]
        to_tensor = transforms.ToTensor()
        for image_path in images_paths:
            image = Image.open(image_path)
            if transform:
                image = transform(image)
            image = to_tensor(image)
            self.images.append(image)

        self.images = torch.stack(self.images)
        self.len = len(self.images)

    def __len__(self):
        return self.len

def getSupportSets(dataset_dir, class_label_map_json, transform=None):
    sets = {}

    with open(class_label_map_json, 'r') as f:
        class_label_map = json.load(f)

    for class_name, attributes in class_label_map.items():
        label = attributes['label']
        cro_name = attributes['cro_name']
        description = attributes['description']
        
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            support_dir = os.path.join(class_dir, "support")
            sets[label] = SupportSet(support_dir, class_name, label, cro_name, description, transform)
    return sets

def getSupportSetsEmbeddings(model, support_sets, device):
    embeddings = {}

    for label in support_sets.keys():
        embeddings[label] = torch.mean(model(support_sets[label].images.to(device)), dim=0)                #torch.Size([set_len, 256]) => #torch.Size([256])

    return embeddings

class SiameseNetwork(nn.Module):
    def __init__(self, feature_extractor, threshold, support_sets, device):
      super(SiameseNetwork, self).__init__()
      self.feature_extractor = feature_extractor
      self.threshold = threshold
      self.support_sets = support_sets
      self.device = device
      self.support_sets_embeddings = getSupportSetsEmbeddings(self.feature_extractor, self.support_sets, self.device)
      self.embeddings_support = torch.stack([self.support_sets_embeddings[label] for label in self.support_sets.keys()]) #torch.Size([cls_num, 256])

    def forward(self, x):
        # If input tensor has 3 dimensions, add a batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        embeddings = self.feature_extractor(x)  # torch.Size([img_num, 256])

        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_support = F.normalize(self.embeddings_support, p=2, dim=1)

        cosine_similarities = torch.matmul(embeddings, embeddings_support.t())  # torch.Size([img_num, cls_num])

        # Cosine similarity ranges from -1 to 1
        _, max_class_indices = cosine_similarities.max(dim=1)

        predictions = [max_class_indices[i].item() for i in range(x.size(0))]

        return predictions

class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.base_model = base_model
        self.flatten = nn.Flatten(1,-1)
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = F.normalize(x)
        return x

feature_extractor = torch.load('model/feature_extractor.pth', map_location=torch.device('cpu'))

siamese_network = torch.load('model/siamese_network.pth', map_location=torch.device('cpu'))

support_sets_embeddings = torch.load('model/support_sets_embeddings.pth', map_location=torch.device('cpu'))

def preprocess_image(image):
    image = image.resize((180, 180))
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    return image


def predict(image):
    processed_image = preprocess_image(image)
    processed_image = processed_image.to(torch.float32)

    prediction = siamese_network(processed_image)
    print(prediction)
    return prediction


classes = getSupportSets('iNaturalist dataset', 'label_map.json')

st.title("Klasifikacija rijetkih biljaka i životinja")

classification_method = st.selectbox(
    'Odaberite model za klasifikaciju',
    ('Swin_V2 - najbolji za klasifikaciju životinja', 'ResNet50 - najbolji za klasifikaciju biljaka', 'MyFSL')
)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if st.button('Process Image'):
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        prediction_label = predict(image)

        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        st.text("Processing the image...")

        result = f"Predpostavljena vrsta je {classes[prediction_label[0]].class_name}\nNaziv: {classes[prediction_label[0]].cro_name}\n{classes[prediction_label[0]].description}"
        
        st.text(result)
    else:
        st.text("Please upload an image before clicking the button.")
