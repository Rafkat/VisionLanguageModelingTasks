import torch
from torch import nn
from transformers import ViTModel, LlamaModel


# originated from https://arxiv.org/pdf/2301.12597


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.encoder = LlamaModel.from_pretrained('huggyllama/llama-7b')
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


class CrossModelTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, blocks_num, dropout=0.1):
        super(CrossModelTransformer, self).__init__()
        pass


class QFormer(nn.Module):
    def __init__(self, vision_embed_dim, text_embed_dim, num_heads, blocks_num, hidden_size, dropout=0.1):
        super(QFormer, self).__init__()
        self.queries = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.vision_projection = nn.Linear(vision_embed_dim, hidden_size)
        self.text_projection = nn.Linear(text_embed_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_modal_transformer = CrossModelTransformer(hidden_size, num_heads, blocks_num, dropout=dropout)

    def forward(self, image_embeddings, text_embeddings):
        image_proj = self.vision_projection(image_embeddings)
        text_proj = self.text_projection(text_embeddings)
        x = self.cross_modal_transformer(image_proj, text_proj, self.queries)
        return x


class BLIP2(nn.Module):
    def __init__(self, vision_embed_dim, text_embed_dim, num_heads, blocks_num, hidden_size, vocab_size, dropout=0.1):
        super(BLIP2, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.qformer = QFormer(vision_embed_dim, text_embed_dim, num_heads, blocks_num, hidden_size, dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, texts):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(texts)
        x = self.qformer(image_embeddings, text_embeddings)
        x = self.fc(x)
        return x
