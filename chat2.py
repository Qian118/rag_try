from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import numpy as np
import pickle
import json
import torch
device = torch.device("cuda:0") 
with open("train_similar_model/id_vector_zhu","rb") as f:
    faiss_index=pickle.load(f)
model_path="train_similar_model/my_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
recall_model = BertModel.from_pretrained(model_path)
rank_model = BertForSequenceClassification.from_pretrained('train_similar_model/rank_model')

chat_tokenizer = AutoTokenizer.from_pretrained("/media/zhjk/zwb_dir/Chatglm3/", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/media/zhjk/zwb_dir/Chatglm3/", trust_remote_code=True,device_map="balanced_low_0").cuda()
#chat_model=chat_model.to(device)

rank_model=rank_model.to(device)
recall_model=recall_model.to(device)
def get_similar_query(query,num=3):
    results=[]
    for _ in range(0,num):
        #大模型进行改写
        #do_sample=True 随机采样，不然每次采样的内容都一样。丧失多样性
        response, _ = chat_model.chat(chat_tokenizer, query+"。你的任务是改写上述语句，表达相似的语义，不需要拓展内容", history=[],do_sample=True,num_beams=3,temperature=0.8)
        
        results.append(response)
    return results

#问题改写，进行招呼，都经过rank打分模型，

def read_knowledge(path):
    with open(path,encoding="utf-8") as f:
        lines=f.readlines()
    data=[json.loads(line.strip()) for line in lines]
    id_desc={}
    for s in data:
        id=s['id']
        id_desc[id]=s
    return id_desc
def normal(vector):
    vector=vector.tolist()[0]
    ss=sum([s**2 for s in vector])**0.5
    return [s/ss for s in vector]
def get_vector(sentence):
    encoded_input = tokenizer([sentence], padding=True, truncation=True,return_tensors='pt')
    encoded_input=encoded_input.to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = recall_model(**encoded_input)
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = normal(model_output[1])
    sentence_embeddings=np.array([sentence_embeddings])
    return sentence_embeddings
def get_candidate(input,num=20):
    #转项量，然后通过向量获取候选
    vector=get_vector(input)
    D, I = faiss_index.search(vector, num)
    D=D[0]
    I=I[0]
    indexs=[]
    for d,i in zip(D,I):
        indexs.append(i)
    return indexs

def rank_sentence(query,sentences):
    #sentences召回的内容
    #query 用户的问题
    X=[[query[0:200],sentence[0:200]] for sentence in sentences]
    X = tokenizer(X, padding=True, truncation=True, max_length=512,return_tensors='pt')
    X=X.to(device)
    scores=rank_model(**X).logits #二分类
    scores=torch.softmax(scores,dim=-1).tolist()
    scores=[round(s[1],3) for s in scores] #将其四舍五入到小数点后三位，得到一个分数，1分相似度高，0分相似度无
    return scores
def rag_recall(query):
    #query 用户的问题
    #对用户输入的句子，使用大模型产生相似的句子
    similar_querys=get_similar_query(query)
    #if len(similar_querys) < 3:
      #query = input("请详细描述您的症状：")
      #similar_querys=get_similar_query(query)
    print('---->\n',similar_querys)
    index_score={}
    for input1 in [query]+similar_querys:
        #对于每一个query，拿num个候选，通过faiss的索引，快速获取
        indexs=get_candidate(input1,num=30)
        sentences=[id_knowledge[index]['病情描述'] for index in indexs]
        #rank在模型层面上没有交互，我们需要进行精排，精排也是bert模型，
        #只在召回后的数据中进行计算
        scores=rank_sentence(input1,sentences)
        for index,score in zip(indexs,scores):
            #低于0.9的分数都过滤
            if score<0.9:
                continue
            #计算各个候选的得分   要么召回得分高，要么召回次数多，
            index_score[index]=index_score.get(index,0.0)+score

    results=sorted(index_score.items(),key=lambda s:s[1] ,reverse=True)
    #取前三个得分最高的
    return results[0:3]
def get_prompt(recall_result):

    prompt=""
    #知识的id，召回的分数
    for i,[recall_id,recall_score] in enumerate(recall_result):
        prompt+="案例{}：".format(i)+"病情描述："+id_knowledge[recall_id]['病情描述']+"治疗方案:"+id_knowledge[recall_id]['治疗方案']+"。"
    return prompt
 


id_knowledge=read_knowledge("knowledge_zhu")

while True:
    query= input("输入症状: ")
    recall_result=rag_recall(query)
    #参考经验 用户咨询，-医生答复
    prompt=get_prompt(recall_result)
    print(prompt)
    response, _ = chat_model.chat(chat_tokenizer, prompt+"根据上述治疗方案，给出下述病情的治疗方案"+query,history=[])
    print ('--->\n',response)
# similar_querys=get_similar_query(query)
# print (similar_querys)
# sentences=get_candidate(input,num=30)
# scores=rank_sentence(input,sentences)
# results=list(zip(sentences,scores))
# results=sorted(results,key=lambda s:s[1] ,reverse=True)
# print (results)
 

