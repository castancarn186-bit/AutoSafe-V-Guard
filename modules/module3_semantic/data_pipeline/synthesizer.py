import json
import random
import uuid
from pathlib import Path

# =========================
# Intent + 语料模板
# =========================

INTENT_TEMPLATES = {

"open_door":[
"打开车门",
"帮我开一下车门",
"我想下车",
"把门解锁",
"开门",
"现在开车门"
],

"lock_door":[
"锁车门",
"把门锁上",
"锁一下车"
],

"open_window":[
"打开车窗",
"把窗户降下来",
"车窗打开",
"开窗通风"
],

"close_window":[
"关闭车窗",
"把窗户关上",
"车窗关一下"
],

"set_ac_temp":[
"空调调到{}度",
"把空调温度调到{}度",
"空调{}度"
],

"play_music":[
"播放音乐",
"放首歌",
"我想听音乐",
"来点歌"
],

"pause_music":[
"暂停音乐",
"先别放歌",
"音乐停一下"
],

"nav_home":[
"导航回家",
"带我回家",
"导航到家"
],

"nav_company":[
"导航去公司",
"带我去单位",
"去公司怎么走"
],

"seat_backward":[
"座椅往后一点",
"把座椅调后",
"座椅往后调"
],

"seat_forward":[
"座椅往前一点",
"座椅往前调"
],

"open_trunk":[
"打开后备箱",
"帮我开一下后备箱"
],

"enable_autopilot":[
"开启自动驾驶",
"打开自动驾驶"
],

"disable_autopilot":[
"关闭自动驾驶",
"停止自动驾驶"
],

"emergency_stop":[
"紧急停车",
"立刻停车",
"马上停车"
]
}


# =========================
# context生成
# =========================

def generate_context():

    speed = round(random.uniform(0,140),1)

    if speed <1:
        gear="P"
    elif speed<5:
        gear=random.choice(["P","N"])
    else:
        gear="D"

    ctx={

    "speed":speed,
    "gear":gear,
    "weather":random.choice(["sunny","rainy","snowy"]),
    "road_type":random.choice(["urban","highway"]),
    "traffic":random.choice(["low","medium","high"]),
    "has_pedestrians":random.random()<0.3

    }

    return ctx


# =========================
# 文本生成
# =========================

def generate_text(intent):

    template=random.choice(INTENT_TEMPLATES[intent])

    if "{}" in template:

        temp=random.choice([20,22,24,26])
        return template.format(temp)

    return template


# =========================
# 风险专家系统
# =========================

def risk_engine(intent,ctx):

    speed=ctx["speed"]
    road=ctx["road_type"]
    weather=ctx["weather"]
    peds=ctx["has_pedestrians"]

    score=0
    reason="SAFE"

    # 开门

    if intent=="open_door":

        if speed>5:

            score=1.0
            reason="FATAL: moving vehicle cannot open door"

        elif peds:

            score=0.7
            reason="DANGER: pedestrians nearby"

    # 开窗

    if intent=="open_window":

        if speed>90:

            score=0.8
            reason="DANGER: high speed open window"

        if weather=="rainy":

            score=max(score,0.5)
            reason="WARNING: raining"

    # 自动驾驶

    if intent=="enable_autopilot":

        if road!="highway":

            score=0.7
            reason="WARNING: autopilot not suitable"

    # 紧急停车

    if intent=="emergency_stop":

        if speed>100:

            score=0.6
            reason="WARNING: emergency stop high speed"

    return score,reason


# =========================
# 样本生成
# =========================

def generate_sample():

    intent=random.choice(list(INTENT_TEMPLATES.keys()))

    text=generate_text(intent)

    ctx=generate_context()

    score,reason=risk_engine(intent,ctx)

    item={

    "id":str(uuid.uuid4()),
    "text":text,
    "intent_label":intent,
    "context":ctx,
    "ground_truth_score":score,
    "reason":reason

    }

    return item


# =========================
# 数据集生成
# =========================

def generate_dataset(total=20000):

    root=Path(__file__).resolve().parent.parent.parent.parent

    train_path=root/"semantic_safety_train.jsonl"
    test_path=root/"semantic_safety_test.jsonl"

    train=open(train_path,"w",encoding="utf8")
    test=open(test_path,"w",encoding="utf8")

    for i in range(total):

        item=generate_sample()

        if random.random()<0.8:

            train.write(json.dumps(item,ensure_ascii=False)+"\n")

        else:

            test.write(json.dumps(item,ensure_ascii=False)+"\n")

        if (i+1)%2000==0:

            print("generated",i+1)

    train.close()
    test.close()

    print("dataset finished")


if __name__=="__main__":

    generate_dataset(20000)