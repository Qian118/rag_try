disease_desc={}
all_desc=[]
data = [
    {"id": 0, "病情描述": "强制性脊柱炎，晚上睡觉翻身时腰骶骨区域疼痛，其他身体任何部位均不疼痛。", "治疗方案": "应该没有问题，但最好把图像上传看看。"},
    {"id": 1, "病情描述": "先天性髋关节发育不良，半脱位.，右侧髋和膝疼痛不能随意行走.拍过x片,为先天性髋关节发育不良，半脱位.", "治疗方案": "谢谢你的问题，门诊我们见到了，有事电话联系。"}
]
for s in data:
    desc=s["病情描述"]
    disease="强制性脊柱炎"
    if disease not in disease_desc:
        disease_desc[disease]=[]
    disease_desc[disease].append(desc)
    all_desc.append(desc)