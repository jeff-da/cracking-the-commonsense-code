from bert_embedding import BertEmbedding
import mxnet as mx

phrases = """airplane is chewy"""
sentences = phrases.split('\n')
bert_embedding = BertEmbedding()
result = bert_embedding(sentences, 'avg', False)
for i, item in enumerate(result):
    print(item[0])
    # if i == 0:
    #     print(item[1])