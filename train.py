import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import SGD,Adagrad,Adam,RMSprop

from bert_tokenizers import Tokenizer
from bert_models import Bert,build_model
from utils import bert_params,model_params,HingeLoss,MeanAveragePrecision
from data import DataGenerator
from models import Bert4QA

import time

class TrainOrPredict(object):
    
    def __init__(self, model_params, ckpt_path = './save_checkpoint/',**kwargs):

        self.mp = model_params
        self.ckpt_path = ckpt_path

    def train(self, model, optimizer, datagen, load_ckpt = False):
        '''datagen是数据迭代器,load_ckpt控制是否导入上次的数据'''
        ckpt = tf.train.Checkpoint(model = model, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,self.ckpt_path ,max_to_keep = 2)

        #这里最后再确认下，或者更改参数为batch_size
        num_neg = self.mp.num_neg
        if ckpt_manager.latest_checkpoint and load_ckpt:
            ckpt.restore(ckpt_manager.last_checkpoint)
            print("Load last checkpoint restore")
        print('self.mp.epoch:',self.mp.epoch)
        for epoch in range(self.mp.epoch):
            for step, (left,right,targets) in enumerate(datagen.generate_batch_data()):
                inputs = [left,right]
                
                #这一部分可作为train_one_step
                with tf.GradientTape() as tape:
                    logits = model(inputs)
                    hinge_loss = HingeLoss(logits, targets, num_neg)
                    
                    loss = tf.math.reduce_sum(hinge_loss)
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                    map_value = MeanAveragePrecision(targets,logits)

                #这里是每个epoch都会打印当前效果
                if (epoch + 1)%1 == 0:
                    map_value = MeanAveragePrecision(targets,logits)
                    #print()

                    print('\n'+'-'*50)
                    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
                    print('epoch:{}, step:{}, hinge_loss:{:.4f}, map_value:{:.4f}'.format(epoch+1, step+1,loss.numpy(), map_value.numpy()))
                    print('-'*50+'\n')
                    ckpt_manager.save()
        return model
    
    def predict(self,model,optimizer,inputs):
        pass

def main():

    #模型训练预测时的参数
    mp = model_params()

    #构建bert的参数
    bp = bert_params(with_pool = True)

    datagen = DataGenerator(bp, batch_size = mp.batch_size, num_neg = mp.num_neg, shuffle = mp.shuffle)

    #后续再尝试用其它的优化器
    optimizer = Adagrad(learning_rate = mp.learning_rate)
    my_model = Bert4QA(bp)
    
    #训练主类
    t = TrainOrPredict(mp)

    #final_model就是训练好的模型
    final_model = t.train(my_model,optimizer,datagen)

    data = datagen.data_faq
    tokenizer = datagen.tokenizer

    #训练完成后查看效果
    real_query_text = "月球和地球是什么关系?"
    
    question_score = {}
    for query_name in data.query_dict.keys():
        query_text = data.query_dict[query_name]
        token_ids,segment_ids = tokenizer.encode(real_query_text,query_text)
        question_score[query_name] = final_model.predict([token_ids,segment_ids])
    question_score={k:v.numpy() for k,v in question_score.items()}
    qs=dict(sorted(question_score.items(),key=lambda x:x[1],reverse=True))
    c=0
    for k,v in qs.items():
        c+=1
        print(k,data.query_dict[k],v)
        if c==10:break

    return final_model
    
if __name__ == "__main__":

    final_model = main()

