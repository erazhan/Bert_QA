import tensorflow as tf
import numpy as np

#bert_layer模型的一些参数
class bert_params:

    def __init__(self,with_pool = False,with_nsp = False, with_mlm = False):
        self.vocab_size=21128
        self.hidden_size=768
        self.num_hidden_layers=12
        self.num_attention_heads=12
        self.intermediate_size=3072
        self.hidden_act='gelu'
        self.max_position_embeddings=512
        self.hidden_dropout_prob=0.1
        self.attention_probs_dropout_prob=0.1

        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

        self.bert_dir = './data_or_model/chinese_L-12_H-768_A-12/'
        self.bert_config = self.bert_dir + 'bert_config.json'
        self.bert_ckpt = self.bert_dir + 'bert_model.ckpt'
        self.bert_vocab = self.bert_dir + 'vocab.txt'

        #以下是QA参数部分
        self.UA_file_path = "./data_or_model/qadata/UserAsked.txt"
        self.FAQ_file_path = "./data_or_model/qadata/FAQ.txt"
        self.PN_file_path = "./data_or_model/qadata/PosNeg.txt"

#训练预测时调整的参数
class model_params:
    epoch = 10

    optlist=['adam','adagrad','sgd','rmsprop']
    optimizer = optlist[0]
    
    #学习率尽量小
    learning_rate = 0.0005
    batch_size = 8
    num_dup = 1
    num_neg = 1
    
    #取每个epoch时混洗数据
    shuffle = True
    
    #命名后续调整
    #embed_units = 256

    #这个用bert模型中的
    #vocab_size = 266
    #dropout = 0.3

    #remove_checkpoint = False

#没必要写成一个Layer
def HingeLoss(y_pred,y_true,num_neg):
    '''
    #其实命名为y_pred和y_true会理解错
    y_pred,y_true的维度:(batch_size*(num_neg+1),1)
    num_neg:负采样个数
    loss的维度(batch_size,1)
    '''
    
    batch_size = tf.cast(tf.shape(y_pred)[0]/(num_neg+1),dtype=tf.int32)

    #注意这里的维度一定要写成(batch_size,1),写成(batch_size,)是有问题的
    neg_pred = tf.zeros((batch_size,1),dtype=tf.float32)
    pos_pred = tf.cast(y_pred[::(num_neg+1)],dtype=tf.float32)
    
    for i in range(num_neg):
        tem = tf.cast(y_pred[(i+1)::(num_neg+1)],dtype=tf.float32)
        
        neg_pred = tf.math.add(neg_pred,tem)
        
    #要做平均,否则正例的预测值直接非常大
    neg_pred = neg_pred/num_neg
    
    loss = tf.maximum(0.,5 + neg_pred - pos_pred)
        
    return loss

def MeanAveragePrecision(y_true,y_pred,threshold=0.0):

    #threshold=tf.cast(threshold,dtype=tf.float32)

    y_pair=list(zip(tf.cast(y_true,dtype=tf.float32),tf.cast(y_pred,dtype=tf.float32)))
    y_pair=sorted(y_pair, key=lambda x: x[1], reverse=True)
    
    #pos记录依次预测正确的正例个数，map_value记录map值
    pos,map_value=0,0.0
    
    for idx,(yt,yp) in enumerate(y_pair):
        if yt>threshold:
            pos+=1
            map_value+=pos/(idx+1.0)
    if pos == 0:
        return 0.
    else:
        return tf.convert_to_tensor(map_value/pos,dtype=tf.float32)
