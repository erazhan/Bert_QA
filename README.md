# Bert_QA
基于Bert的信息检索问答

# 内容介绍
本项目利用Bert实现了一个简单的基于信息检索的问答系统，也就是给定query，遍历所有的question(或answer)计算评分，按评分从高到低进行返回结果。

## 说明
save_checkpoint.zip保存了模型的checkpoint文件，使用时重新训练生成即可。data_or_model文件夹中包含两个文件(夹)，qadata和chinese_L-12_H-768_A-12.zip。

### qadata
包含3个文件，可以将具体的数据按下面的格式进行收集存储，然后调整好文件路径等参数就可以直接使用。另外构造负例的一个技巧是将相似的query设置为负例。
- FAQ.txt
存储标准问答对，文件格式是:question_name question_text answer_name answer_text，每个问答对占一行，中间有3个空格符分隔
- UserAsked.txt
存储用户query，文件格式是:query_name query_text，
- PosNeg.txt
对每个query构造正负例question，文件格式是:query_name pos_question_name neg_question_name,分别代表query，正确question，错误question(可多个)

### chinese_L-12_H-768_A-12.zip
google官方预训练好的中文版Bert参数，使用时解压到同名文件夹下使用

# 代码参考
[bojone/bert4keras](https://github.com/bojone/bert4keras)
