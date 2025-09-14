import streamlit as st
import  pandas as pd
from streamlit import button
from utils import dataframe_agent

def create_chart(input_data, chart_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    # 需要设置dataframe索引 把第一列设置为索引
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)

st.title("💡 CSV数据分析智能工具")

with st.sidebar:
    opeenai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-keys)")

data = st.file_uploader("上传你的数据文件（CSV格式）：", type="csv")
if data:
    # 把CSV读取为DataFrame 并作为变量储存进对话状态里
    st.session_state["df"] = pd.read_csv(data)
    with st.expander("原始数据"):
        st.dataframe(st.session_state["df"])

query = st.text_area("请输入你关于以上表格的问题，或数据提取请求，或可视化要求（支持散点图、折线图、条形图）：")
button = st.button("生成回答")

if button and not opeenai_api_key:
    st.info("请输入你的OpenAI API密钥")
if button and "df" not in st.session_state:
    st.info("请先上传数据文件")
if button and opeenai_api_key and "df" in st.session_state:
    with st.spinner("AI正在思考中，请稍等..."):
        response_dict = dataframe_agent(opeenai_api_key, st.session_state["df"], query)
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        if "table" in response_dict:
            # .table函数是用来展示表格组件的 第一个参数是表格里的内容
            st.table(pd.DataFrame(response_dict["table"]["data"],
            # .table函数接受dataframe作为参数 通过pandas的DataFrame进行类型的转换 dataframe的参数就是表格里的内容
                                  columns=response_dict["table"]["columns"]))
        if "bar" in response_dict:
            create_chart(response_dict["bar"], "bar")
        if "line" in response_dict:
            create_chart(response_dict["line"], "line")
        if "scatter" in response_dict:
            create_chart(response_dict["scatter"], "scatter")