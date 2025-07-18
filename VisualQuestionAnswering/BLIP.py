import numpy as np
from torch import nn
import torch
from transformers import ViTModel, BertModel, BertLMHeadModel, BertTokenizer


# originated from https://arxiv.org/pdf/2201.12086

class BLIP(nn.Module):
    def __init__(self):
        super(BLIP, self).__init__()

        self.visual_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.tokenizer = BertTokenizer.from_pretrained('google/bert-base-cased')
        self.text_encoder = BertModel(add_pooling_layer=False).from_pretrained('google-bert/bert-base-uncased')
        self.text_decoder = BertLMHeadModel.from_pretrained('google-bert/bert-base-uncased')

    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                  return_tensors='pt').to(image.device)
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        if train:
            answer = self.tokenizer(answer, padding='longest', return_tensors='pt').to(image.device)
            answer.input_ids[:, 0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            question_states = []
            question_atts = []
            for b, n in enumerate(n):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [question.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none')

            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)
            return loss
        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            if inference == 'generate':
                num_beams = 3
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
                question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(question_states.device)
                model_kwargs = {'encoder_hidden_states': question_states, 'encoder_attention_mask': question_atts}

                bos_ids = torch.full((image.size(0), 1), fill_value=self.tokenizer.bos_token_id,
                                     device=image.device)
                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id,
                                                     **model_kwargs)
                answers = []
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)
                    answers.append(answer)
                return answers
            elif inference == 'rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                           answer.input_ids, answer.attention_mask, k_test)
                return max_ids

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]

        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))

        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        target_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        question_states = self.tile(question_states, 0, k)
        question_atts = self.tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=target_ids,
                                   return_dict=True,
                                   reduction='none')

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
        return max_ids

    @staticmethod
    def tile(x, dim, n_tile):
        init_dim = x.size(dim)
        repeat_idx = [1] * x.dim()
        repeat_idx[dim] = n_tile
        x = x.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(x, dim, order_index.to(x.device))
