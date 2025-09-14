from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import json

# 没用langchain.prompt的PromptTemplate.from_template是因为create_pandas_dataframe_agent已经生成了一个框架极内置prompt（ReAct协议）
# PROMPT_TEMPLATE只是应用层的约束 而Executor只认框架极协议（Action/Final Answer）
# 我的JSON约束只有在被包装到Final Answer里才有效
# 如果自己用.from_template 相当于绕过agent直接和模型交互 必须自己写Action/Final Answer这一整套协议 甚至还得自己实现executor的逻辑（循环 解析 错误反馈）
# agent = Prompt + Executor + 工具管理
PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。

1. 对于文字回答的问题，按照这样的格式回答：
   {"answer": "<你的答案写在这里>"}
例如：
   {"answer": "订单量最高的产品ID是'MNWC3-067'"}

2. 如果用户需要一个表格，按照这样的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用户的请求适合返回条形图，按照这样的格式回答：
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用户的请求适合返回折线图，按照这样的格式回答：
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用户的请求适合返回散点图，按照这样的格式回答：
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。


请将所有输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}

你要处理的用户请求如下： 
"""
# 不管被响应的格式被要求成什么样 AI返回的都是字符串 下一步才是解析为字典
# 因此要求AI返回JSON字符串 便于解析
# 变量命名是小写 类命名是驼峰 常量命名是全大写(约定只读)


def dataframe_agent(openai_api_key, df, query):
# 第二个参数表示一个表格 以DataFrame作为具体的类型 DataFrame是pandas的特殊类
    model = ChatOpenAI(model="gpt-4-turbo",
                       openai_api_key=openai_api_key,
                       temperature=0)
    # 定义一个agent执行器
    # 定义agent的时候传入的prompt是修改【框架层】协议的prompt
    agent = create_pandas_dataframe_agent(llm=model,df=df,
                                          agent_executor_kwargs={"handle_parsing_errors": True},
                                          # 尽可能让模型自行消化和处理错误 而不是让程序直接终止
                                          verbose=True)
                                          # 为了了解模型是如何思考的 可以在终端看到程序执行过程

    # 如果只是想获得AI针对DataFrame的回答 可以把query直接传给agent的invoke方法 它返回的内容会直接作为字符串展示
    # 但该项目的前端需要知道该把返回的内容直接展示为字符串 还是展示为图表
    # 一个解决办法是让agent返回一个字典 根据字典里的不同键 来判断下一步操作
    # 比如字典包含answer键->展示字符串 包含table键->展示图表 那么要把这个要求在提示中写清楚 详见上面Template

    # 把用户的要求和补充的要求拼接起来
    prompt = PROMPT_TEMPLATE + query
    response = agent.invoke({"input": prompt})
    # 调用时传入的prompt是【应用层】prompt
    response_dict = json.loads(response["output"])
    return response_dict

# import os
# import pandas as pd
#
# df = pd.read_csv("personal_data.csv")
# print(dataframe_agent(os.getenv("OPENAI_API_KEY"), df, "数据量出现最多的职业是什么？"))