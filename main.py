# main.py
import sys
import os
from dotenv import load_dotenv
from models.Classification import Classification
from Divide_Columns import ColumnDivider
from models.Prediction import Prediction  
load_dotenv()

def main():

    while True:
        print("请选择功能：1-列功能 2-AI分类模块 3-AI预测原因模块  4-退出")  # 修改
        choice = input("输入选择(1/2/3/4)：").strip()  # 修改
        if choice == "1":
            file_path = os.getenv("FILE_PATH") 
            divider = ColumnDivider(file_path)
            divider.read_excel()
            divider.add_recommend_column()
            divider.add_reason_column()
            output_path = divider.save_excel()
            print(f"处理完成，文件已保存至：{output_path}")
        elif choice == "2":  # 这个是2,客户原因归类    
            file_path = os.getenv("FILE_PATH2")
            output_file=os.getenv("OUTPUT_FILE")
            analyzer = Classification(file_path,output_file)
            analyzer.process_and_analyze()
            print("AI分析完成，已更新Excel文件")
        elif choice == "3":  # 这个是1,AI预测原因
            file_path = os.getenv("FILE_PATH2")
            output_file=os.getenv("OUTPUT_FILE")
            analyzer = Prediction(file_path,output_file)
            analyzer.process_and_analyze()
            print("分析完成完成")
        elif choice == "4":
            print("退出")
            break
        else:
            print("无效选择")

if __name__ == "__main__":
    main()
