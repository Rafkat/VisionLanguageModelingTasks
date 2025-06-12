import torch
from torch import nn
from torchvision import models
from transformers import DistilBertModel
import torch.nn.functional as F
from PIL import Image


# originated from https://arxiv.org/pdf/2103.00020

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, **kwargs):
        self.image_filenames = kwargs.get('image_filenames', None)
        self.captions = kwargs.get('captions', None)
        tokenizer = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.encoded_captions = tokenizer(self.captions, padding=True, truncation=True, max_length=200)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = Image.open().convert('RGB')
        image = self.transforms(image)
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        pretrained_model = models.resnet50(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(pretrained_model.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = True

        self.target_token_idx = 0

    def forward(self, x, attention_mask):
        output = self.model(input_ids=x, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self, temperature=1.0, image_embedding_dim=2048, text_embedding_dim=768, projection_dim=256,
                 dropout=0.1):
        super(CLIPModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(image_embedding_dim, projection_dim, dropout=dropout)
        self.text_projection = ProjectionHead(text_embedding_dim, projection_dim, dropout=dropout)
        self.temperature = temperature

    def forward(self, image, text, text_mask):
        image_features = self.image_encoder(image).squeeze()
        text_features = self.text_encoder(text, text_mask)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        logits = text_embeddings.matmul(image_embeddings.T) / self.temperature
        images_similarity = image_embeddings.matmul(image_embeddings.T)
        texts_similarity = text_embeddings.matmul(text_embeddings.T)
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy_loss(logits, targets, reduction='none')
        images_loss = self.cross_entropy_loss(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()

    @staticmethod
    def cross_entropy_loss(logits, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(logits)).sum(dim=1)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()


if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    inputs_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    model = CLIPModel()
    model(images, inputs_ids, attention_mask)
