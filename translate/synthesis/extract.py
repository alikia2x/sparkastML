from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
text='''互联
虎脸
互怜
糊脸对猴
互联工程
互联互通
湖莲潭
互联网
互联网安全
互联网编程
互联网产品
互联网出版管理暂行规定
互联网创业
互联网大会
互联网等信息网络传播视听节目管理办法
互联网电脑
互联网服务
互联网公司'''
messages = [
     {"role": "system", "content": "用户会给出若干中文短语或词汇，每行一个。你需要从中抽取出**不重复**的中文**词汇**并输出，每行一个。**注意，你不应该输出其它任何内容**"},
     {"role": "user", "content": text},
]
response = client.chat.completions.create(model='deepseek-v2',messages=messages,temperature=1.0)
print(response.choices[0].message.content)