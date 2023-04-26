# BERT(title) + CodeBERT(code) + BERT(desc)
MAX_DESC_LEN = 512
MAX_TITLE_LEN = 64
MAX_CODE_LEN = 256

from torch import nn, tensor, cat
from transformers import AutoTokenizer, AutoModel

class TagRecommandModel(nn.Module):
    def __init__(self, tags_count, freeze_bert_title=False, freeze_bert_code=False, freeze_bert_desc=False, hidden_dropout_prob=0.1):
        super(TagRecommandModel, self).__init__()
        print('加载模型')
        self.tokenizer_title = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer_code = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.tokenizer_desc = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.bert_title = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_code = AutoModel.from_pretrained('microsoft/codebert-base')
        self.bert_desc = AutoModel.from_pretrained('bert-base-uncased')

        if freeze_bert_title:
            for param in self.bert_title.parameters():
                param.requires_grad = False
        if freeze_bert_code:
            for param in self.bert_code.parameters():
                param.requires_grad = False
        if freeze_bert_desc:
            for param in self.bert_desc.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(768 * 3, tags_count)
        # self.dense.weight.data.normal_(0, 0.01)
        self.sigmoid = nn.Sigmoid()

    def part_embedding(self, tokenizer, bert, batch_sentences, length):
        encoded = tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, truncation=True, max_length=length, padding='max_length')
        tokens_ids = tensor(encoded['input_ids']).cuda()
        attention_mask = tensor(encoded['attention_mask']).cuda()
        bert_result = bert(input_ids=tokens_ids, attention_mask=attention_mask)
        # embeddings = bert_result[0][:,0,:].contiguous()
        embeddings = bert_result[1]
        return embeddings
    
    def forward(self, batch_title, batch_code, batch_desc):
        embeddings_title = self.part_embedding(self.tokenizer_title, self.bert_title, batch_title, MAX_TITLE_LEN)
        embeddings_code = self.part_embedding(self.tokenizer_code, self.bert_code, batch_code, MAX_CODE_LEN)
        embeddings_desc = self.part_embedding(self.tokenizer_desc, self.bert_desc, batch_desc, MAX_DESC_LEN)

        embeddings = cat((embeddings_title, embeddings_code, embeddings_desc), 1)
        dropout = self.dropout(embeddings)
        dense = self.dense(dropout)
        # print(dense.size())
        # print(dense)
        sigmoid = self.sigmoid(dense)
        return sigmoid, embeddings

if __name__ == '__main__':
    batch_sentences = ['test aaa hahaha', 'var i = 0']
    model = TagRecommandModel(200)
    model = model.cuda()
    print(model(batch_sentences, batch_sentences, batch_sentences))