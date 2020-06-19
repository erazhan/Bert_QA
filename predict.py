import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import SGD,Adagrad,Adam,RMSprop

from bert_tokenizers import Tokenizer
from models import Bert4QA
from data import DataBasic
from utils import model_params,bert_params

#不打印warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test():
    
    mp = model_params()
    bp = bert_params(with_pool = True)
    
    #其实这里内部不用导入一遍bert源码数据
    my_model = Bert4QA(bp)
    
    tokenizer = Tokenizer(bp.bert_vocab,do_lower_case = True)


    optimizer = Adagrad(learning_rate = mp.learning_rate)

    #注意参数名一定要和保存时的参数名一致
    ckpt = tf.train.Checkpoint(optimizer = optimizer,model = my_model)
    ckpt.restore(tf.train.latest_checkpoint("./save_checkpoint/"))
    

    #后面的和导入数据没有关系
    data_faq = DataBasic(bp.FAQ_file_path)
    real_query_text = "月球和地球是什么关系?"
    real_query_text = "月球和地球的关系"
    real_query_text = "月球是地球的卫星吗"
    
    print("实际查询：",real_query_text)
    question_score = {}
    
    for query_name in data_faq.query_dict.keys():
        query_text = data_faq.query_dict[query_name]
        token_ids,segment_ids = tokenizer.encode(real_query_text,query_text)
        question_score[query_name] = my_model.predict([token_ids,segment_ids])

    question_score={k:v.numpy() for k,v in question_score.items()}
    qs=dict(sorted(question_score.items(),key=lambda x:x[1],reverse=True))

    
    for k,v in qs.items():
        
        print(k,data_faq.query_dict[k],v)
        
    return qs
    
if __name__ == "__main__":
    qs = test()
#结果
'''
bert参数导入完成！
实际查询： 月球是地球的卫星吗
Q8 月球和地球是什么关系？ [[-0.7741536]]
Q7 太阳和地球是什么关系？ [[-4.5158362]]
Q1 世界最高的山峰是什么山？ [[-9.345986]]
Q2 世界第二高的山峰是什么山？ [[-9.354628]]
Q5 中国历史的第一位皇帝是谁？ [[-10.089635]]
Q4 黄河的上、中、下游的分界点分别在哪里？ [[-10.298961]]
Q6 中国历史上最后一位皇帝是谁？ [[-10.357178]]
Q3 长江的上、中、下游的分界点分别在哪里？ [[-10.757907]]
'''
