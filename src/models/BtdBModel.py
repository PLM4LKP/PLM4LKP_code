# BERT(title + desc) + BERT(code)
MAX_TEXT_LEN = 512
MAX_CODE_LEN = 256

from torch import nn, tensor, cat
from transformers import AutoTokenizer, AutoModel

class TagRecommandModel(nn.Module):

    def __init__(self, tags_count, freeze_bert_text=False, freeze_bert_code=False, hidden_dropout_prob=0.1):
        super(TagRecommandModel, self).__init__()
        self.embedders = 2
        print('加载模型')
        self.tokenizer_text = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer_code = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.bert_text = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_code = AutoModel.from_pretrained('bert-base-uncased')

        if freeze_bert_text:
            for param in self.bert_text.parameters():
                param.requires_grad = False
        if freeze_bert_code:
            for param in self.bert_code.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(768 * 2, tags_count)
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
        batch_text = [batch_title[i] + ' ' + batch_desc[i] for i in range(len(batch_title))]
        embeddings_text = self.part_embedding(self.tokenizer_text, self.bert_text, batch_text, MAX_TEXT_LEN)
        embeddings_code = self.part_embedding(self.tokenizer_code, self.bert_code, batch_code, MAX_CODE_LEN)

        embeddings = cat((embeddings_text, embeddings_code), 1)
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