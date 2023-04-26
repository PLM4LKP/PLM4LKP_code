# BERT(title + desc)
MAX_TEXT_LEN = 512
MAX_CODE_LEN = 256

from torch import nn, tensor, cat, load
from models.BtdModel import TagRecommandModel

class RelatednessModel(nn.Module):
    def __init__(self, pretrained_model_path, hidden_dropout_prob=0.1):
        super(RelatednessModel, self).__init__()
        print('加载模型')
        self.embedder = TagRecommandModel(23687)
        self.embedder.load_state_dict(load(pretrained_model_path))
        # self.embedder.freeze_bert()

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(768 * 2, 4)
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
    
    def forward(self, batch_title1, batch_code1, batch_desc1, batch_title2, batch_code2, batch_desc2):
        embeddings1 = self.embedder(batch_title1, batch_code1, batch_desc1)[1]
        embeddings2 = self.embedder(batch_title2, batch_code2, batch_desc2)[1]

        embeddings = cat((embeddings1, embeddings2), 1)
        dropout = self.dropout(embeddings)
        dense = self.dense(dropout)
        # print(dense.size())
        # print(dense)
        sigmoid = self.sigmoid(dense)
        return sigmoid, embeddings
        
if __name__ == '__main__':
    batch_sentences = ['test aaa hahaha', 'var i = 0']
    model = RelatednessModel('2022-03-24 145103-epoch1.dat')
    model = model.cuda()
    print(model(batch_sentences, batch_sentences, batch_sentences, batch_sentences, batch_sentences, batch_sentences))