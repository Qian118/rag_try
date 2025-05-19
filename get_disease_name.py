import json
with open("knowledge2",encoding="utf-8") as f:
    lines=[json.loads(s.strip())["疾病名称"] for s in f.readlines()]
name_count={}
for s in lines:
    name_count[s]=name_count.get(s,0)+1
names=[s for [s,v] in name_count.items() if v>50]
 
with open("disease_names","w",encoding="utf-8") as f:
    f.writelines("\n".join(names))