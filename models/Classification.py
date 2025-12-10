# AI_Analyse.py
import json
import os
import time
import pandas as pd
from openai import OpenAI
import openai
from dotenv import load_dotenv  

# 加载环境变量
load_dotenv()  

class Classification:
    def __init__(self,file_name,output_file_name):
        # 从环境变量读取配置
        self.api_key = os.getenv("API_KEY")  
        self.base_url = os.getenv("BASE_URL")  
        self.delay = float(os.getenv("DELAY", "1.0"))
        self.input_excel = f"static/{file_name}"
        self.output_excel=f"static/分类结果/{output_file_name}"
        self.input_json = "static/input.json"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.columns_to_exclude = ['Social是否推荐及理由']
        print("初始化AIAnalyzer完成，配置信息加载成功")

    def excel_to_json(self):
        print(f"开始读取Excel文件: {self.input_excel}")
        df = pd.read_excel(self.input_excel,header=1)
        columns_to_keep = [col for col in df.columns if col not in self.columns_to_exclude]
        df = df[columns_to_keep]
        print(f"Excel文件读取成功，共{len(df)}行数据")
        
        data = df.to_dict(orient="records")
        with open(self.input_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON文件已生成: {self.input_json}")

    def analyze_with_ai(self, content, index):
        print(f"开始分析第{index+1}")
        #print(f"待分析数据: {str(content)[:]}...") 
        #content=content['原因']
        #content=content.loc[:]
        if content['是否推荐']==True:
            system_prompt = {
                "role": "system",
                "content": """你是一个小红书达人总结专家，用户会提供一条和达人相关的信息记录，包含基本信息、客户是否推荐、原因。无论客户有无评论，你都要总结客户 推荐 这位达人的原因，并给出一个包含"原因标签"和"总结说明"的JSON对象。
                        "原因标签"用一个词解释核心原因，只能从[
                                    "数据好",
                                    "进店好",
                                    "内容契合",
                                    "人设契合",
                                    "粉丝匹配",
                                    "展示形式合适",
                                    "剧情植入适合",
                                    "阅读互动强",
                                    "题材匹配",
                                    "风格匹配",
                                    "其它"
                                    ]里面选择。
                                    "总结说明"用一句话详细解释原因，不超过50字。
                                    只返回JSON对象，不添加任何额外文本。"""
            }
        else:
            system_prompt={
                "role": "system",
                "content": """你是一个小红书达人总结专家，用户会提供一条和达人相关的信息记录，包含基本信息、客户是否推荐、原因。无论客户有无评论，你都要总结客户 不推荐 这位达人的原因，并给出一个包含"原因标签"和"总结说明"的JSON对象。
                        "原因标签"用一个词解释核心原因，只能从[
                                    "阅读低",
                                    "数据差",
                                    "数据不稳",
                                    "内容不合适",
                                    "账号类型不合适",
                                    "机构号",
                                    "vlog为主",
                                    "粉丝不匹配",
                                    "价格贵",
                                    "更新少",
                                    "内容水",
                                    "商单表现差",
                                    "破圈账号不符",
                                    "美妆类不符",
                                    "彩妆号",
                                    "粉丝年龄大",
                                    "粉丝年龄小",
                                    "消费力弱",
                                    "人设不符",
                                    "内容杂乱",
                                    "其它"
                                    ]里面选择。
                                    
                                    "总结说明"用一句话详细解释原因，不超过50字。
                                    只返回JSON对象，不添加任何额外文本。"""
            }
        
        user_prompt = {"role": "user", "content": str(content)}
        
        try:
            completion = self.client.chat.completions.create(
                model="qwen-max-latest",
                messages=[system_prompt, user_prompt]
            )
            result = completion.choices[0].message.content.strip()
            result=result.replace("```json\n","")
            result=result.replace("```"," ")
            print(f"AI返回结果: {result}")

            result_json=json.loads(result)
            reason_lab=result_json.get("原因标签")
            summary_description=result_json.get("总结说明")

            time.sleep(self.delay)
            return (reason_lab,summary_description)
        except openai.BadRequestError as e:
            print(f"调用大模型失败: {e}")
            return ""

    def process_and_analyze(self):
        def replace_number(val):
            if val == True:
                return "TRUE"
            else:
                return "FALSE"
        print("开始执行数据处理与AI分析流程")
        self.excel_to_json()
        output_excel=self.output_excel
        print(f"读取JSON文件: {self.input_json}")
        with open(self.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"JSON文件解析完成，共{len(data)}条记录")
        
        print(f"重新读取Excel文件准备添加AI原因列")
        df = pd.read_excel(self.input_excel,header=1)
        
        print(f"开始逐条分析数据...")
        min_length = min(len(df), len(data))
        df = df.iloc[:min_length].reset_index(drop=True)
        data = data[:min_length]

        df["原因标签"]=""
        df["总结说明"]=""
        for i,item in enumerate(data):
            reason_lab,summary_description=self.analyze_with_ai(item,i)
            df.loc[i,"原因标签"]=reason_lab
            df.loc[i,"总结说明"]=summary_description


        #df["AI原因"] = [self.analyze_with_ai(item, i) for i, item in enumerate(data)]
        # df["AI原因"] = [self.analyze_with_ai(
        #         {
        #             "原因": item.get("原因", ""),       
        #             "是否推荐": item.get("是否推荐", "")
        #         }, i
        #     ) for i, item in enumerate(data)
        # ]
        df["是否推荐"].apply(replace_number)
        df.to_excel(output_excel, index=False)
        print(f"\n所有分析完成，已更新Excel文件: {output_excel}")
