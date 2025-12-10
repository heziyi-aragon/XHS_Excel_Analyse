import json
import os
import time
import pandas as pd
from openai import OpenAI
import openai
from dotenv import load_dotenv
from models.Classification import Classification
#import Classificaiton
load_dotenv()

class Prediction(Classification):
    def __init__(self,file_path,output_file):
        super().__init__(file_path,output_file)
        self.input2_json = "static/input2.json"
        self.output_excel = f"static/预测结果/{output_file}"
        self.columns_to_exclude = ['Social是否推荐及理由','原因']

    def excel_to_json(self):
        print(f"开始读取Excel文件并过滤列: {self.input_excel}")
        df = pd.read_excel(self.input_excel,header=1)
        # 过滤掉需要排除的列
        columns_to_keep = [col for col in df.columns if col not in self.columns_to_exclude]
        df_filtered = df[columns_to_keep]
        print(f"过滤后保留{len(columns_to_keep)}列，共{len(df_filtered)}行数据")
        
        data = df_filtered.to_dict(orient="records")
        with open(self.input_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"过滤后的JSON文件已生成: {self.input_json}")
        return data

    def analyze_with_ai(self, content, index):
        print(f"\n开始分析第{index+1}条数据")
        #print(f"待分析数据: {str(content)[:100]}...")
        if content['是否推荐']==True:
            system_prompt = {
                "role": "system",
                "content": """你是一个小红书达人评估专家，用户会提供一条和达人相关的信息记录。你需要评估这位达人值得推荐的原因，并给出一个包含"原因标签"和"总结说明"的JSON对象。
                    "原因标签"用一个词解释核心原因，只能从 [
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
                    ]里面选，
                    "总结说明"用一句话详细解释原因，不超过50字。
                    只返回JSON对象，不添加任何额外文本。"""
            }
        else:
            system_prompt={
                "role": "system",
                "content": """你是一个小红书达人评估专家，用户会提供一条和达人相关的信息记录。你需要评估这位达人不值得推荐的原因，并给出一个包含"原因标签"和"总结说明"的JSON对象。
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

            try:
                result_json = json.loads(result)
                reason_tag = result_json.get("原因标签", "")
                summary = result_json.get("总结说明", "")
            except json.JSONDecodeError as e:
                print(f"解析AI返回的JSON失败: {e}，使用空值")
                reason_tag = ""
                summary = ""
            time.sleep(self.delay)
            return (reason_tag, summary)
        except openai.BadRequestError as e:
            print(f"调用大模型失败: {e}")
            return ("", "")

    def process_and_analyze(self):
        print("开始执行数据处理与AI分析流程")
        filtered_data = self.excel_to_json()
        
        print(f"读取JSON文件: {self.input_json}")
        with open(self.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"JSON文件解析完成，共{len(data)}条记录")
        
        print(f"重新读取Excel文件准备添加AI返回结果列")
        df = pd.read_excel(self.input_excel,header=1)  
        print(f"开始逐条分析数据...")
        
        # 确保数据长度一致
        if len(filtered_data) != len(df):
            print(f"警告：过滤后的数据长度({len(filtered_data)})与Excel数据长度({len(df)})不匹配，将截断较长的部分")
            min_length = min(len(filtered_data), len(df))
            filtered_data = filtered_data[:min_length]
            df = df.iloc[:min_length]
        
        df["原因标签"] = ""
        df["总结说明"] = ""
        
        for i, item in enumerate(filtered_data):
            reason_tag, summary = self.analyze_with_ai(item, i)
            df.loc[i, "原因标签"] = reason_tag
            df.loc[i, "总结说明"] = summary
        
        df.to_excel(self.output_excel, index=False)
        print(f"\n分析完成，已保存至新Excel文件: {self.output_excel}")
        with open(self.input2_json, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        print(f"全部内容已输出至: {self.input2_json}")
