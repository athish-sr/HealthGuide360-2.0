import json 
with open("disease\\diseases_info.json") as f:
    data=json.load(f)
print(len(data))
f.close()
d={}
for k,v in data.items():
    d[k.lower()]=v
with open("disease\\diseases_info.json",'w') as f:
    json.dump(d,f,indent=4)