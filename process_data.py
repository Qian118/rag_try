import json
with open("MedDialog_processed\\train_data.json",encoding="utf-8") as f:
    data=json.load(f)
data=[s for s in data if len(s)==2]
results=[ {"id":i,"病情描述":s[0].replace("病人：",""),"治疗方案":s[1].replace("医生：","")} for i,s in enumerate(data)]
results=[json.dumps(s,ensure_ascii=False) for s in results] #json.dumps 列表变字符串
with open("knowledge_zhu","w",encoding="utf-8") as f:
    f.writelines("\n".join(results))