import csv,unicodecsv

lables = ["天气","温度","湿度","风速","活动"]
dataset = [["晴","炎热","高","弱","取消"]
,["晴","炎热","高","强","取消"]
,["阴","炎热","高","弱","进行"]
,["雨","适中","高","弱","进行"]
,["雨","寒冷","正常","弱","进行"]
,["雨","寒冷","正常","强","取消"]
,["阴","寒冷","正常","强","进行"]
,["晴","适中","高","弱","取消"]
,["晴","寒冷","正常","弱","进行"]
,["雨","适中","正常","弱","进行"]
, ["晴","适中","正常","强","进行"]
,["阴","适中","高","强","进行"]
,["阴","炎热","正常","弱","进行"]
,["雨","适中","高","强","取消"]]
with open('test.csv', 'w', encoding='GBK', newline='') as f:
    writer = csv.writer(f)

    # 写入头
    writer.writerow(list)

    # Python CSV写入多行数据
    writer.writerows(con)

