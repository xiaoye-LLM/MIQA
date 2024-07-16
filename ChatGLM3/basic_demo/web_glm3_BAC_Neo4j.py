import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from neo4j import GraphDatabase  # 确保导入 GraphDatabase
from configBAC import WORD_UNK_ID
from modelBAC import ModelBAC
from utilsBAC import get_vocab, get_label, extract


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'chatglm3-6b')
MODELBAC_PATH = os.path.join(os.path.dirname(__file__), '..', '..', R'BAC\model\best_loss_model2.pth')

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '123456')

# 设置页面标题、图标和布局
st.set_page_config(
    page_title="电机故障诊断智能问答系统",  # 设置页面的标题。在浏览器标签页上显示的名称。
    page_icon=":robot:",  # 设置页面的图标。可以使用 Emoji 表情或者图标的 URL。
    layout="wide"  # 设置页面的布局。在这里设置为 wide，意味着页面会占用更多的水平空间，适合需要展示大量内容的应用程序。
)


#装饰器缓存模型加载过程，以提高应用程序性能。
@st.cache_resource
# 定义获取模型的函数：
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if 'cuda' in DEVICE:
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
        _, word2id = get_vocab()
        id2label, _ = get_label()
        modelBAC = ModelBAC().to(DEVICE)
        modelBAC.load_state_dict(torch.load(MODELBAC_PATH))
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    else:
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()
        _, word2id = get_vocab()
        id2label, _ = get_label()
        modelBAC = ModelBAC().to(DEVICE)
        modelBAC.load_state_dict(torch.load(MODELBAC_PATH))
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return tokenizer, model, modelBAC, word2id, id2label, driver


def entity_identification(question, word2id, id2label):
    # 用于对输入进行命名实体识别
    input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in question]])
    mask = torch.tensor([[1] * len(question)]).bool()
    input = input.to(DEVICE)
    mask = mask.to(DEVICE)
    # 使用BAC模型进行识别
    y_pred = modelBAC(input, mask)
    label = [id2label[l] for l in y_pred[0]]
    info = extract(label, question)
    # 存放知识图谱信息
    Neo4j_data = []  # 知识图谱三元组
    Neo4j_data_str = []  # 知识图谱整理后的自然语言
    # 遍历识别到的实体
    for item in info:
        # 识别出PF实体，并存在与知识图谱
        if 'PF' in item and check_entity_existence(item['PF']):
            Neo4j_data, formatted_results_str = neo4j_query(item['PF'])
            Neo4j_data_str = modify_relationships(formatted_results_str)
            return item['PF'], Neo4j_data_str, Neo4j_data
        # 识别出EQ实体，并存在与知识图谱
        elif 'EQ' in item and check_entity_existence(item['EQ']):
            Neo4j_data, formatted_results_str = neo4j_query(item['EQ'])
            Neo4j_data_str = modify_relationships(formatted_results_str)
            return item['EQ'], Neo4j_data_str, Neo4j_data
    # 返回用户原有输入
    # 测试使用
    # Neo4j_data, formatted_results_str = neo4j_query("转子导条及端环断裂")
    # Neo4j_data_str = modify_relationships(formatted_results_str)
    return question, Neo4j_data_str, Neo4j_data


# 判断实体是否存在于知识图谱中
def check_entity_existence(entity_name):
    with driver.session() as session:
        result = session.run("MATCH (e {name: $entityName}) RETURN COUNT(e) AS count", entityName=entity_name)
        return result.single()['count'] > 0


def modify_relationships(relationships):
    modified_results = []
    for rel in relationships:
        for pattern, replacement in {
            " -[PF_CF]-> ": "故障原因可能是",
            " -[CF_RO]-> ": "故障原因对应的维修操作可能为",
            " -[PF_EQ]-> ": "故障现象可能发生在",
            " -[EQ_PF]-> ": "出现问题可能会导致"
        }.items():
            if pattern in rel:
                rel = rel.replace(pattern, replacement).replace("\n", "")
                break
        modified_results.append(rel)
    return "；".join(modified_results)

def neo4j_query(entity):
    query = "MATCH (e {name: $entity})-[r]->(related) RETURN e.name AS entity, type(r) AS relationship, related.name AS related_entity"
    Neo4j_data = ["电机故障诊断知识图谱相关信息如下：\n"]
    formatted_results_str = []
    with driver.session() as session:
        result = session.run(query, entity=entity)
        if result:
            for record in result:
                relation_str = f"{record['entity']} -[{record['relationship']}]-> {record['related_entity']}\n"
                Neo4j_data.append(relation_str)
                formatted_results_str.append(relation_str)
    return "\n".join(Neo4j_data), formatted_results_str


# 加载要用到的工具
tokenizer, model, modelBAC, word2id, id2label, driver = get_model()

# 初始化历史记录和past key values ，用于存储聊天历史记录和模型的上下文信息。
if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

# 设置max_length、top_p和temperature
st.sidebar.markdown("# 请根据实际情况选择合适的参数")
max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
st.sidebar.markdown("**控制生成文本的最大长度**")
st.sidebar.divider()
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)  # 参数依次为：滑动条的标题，最小值，最大值，初始值，和步长。
st.sidebar.markdown("**较低的top_p值允许更多的低概率此选中，而较高会选择高概率词汇**")  # 参数依次为：滑动条的标题，最小值，最大值，初始值，和步长。
st.sidebar.divider()
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)
st.sidebar.markdown("**较高的温度会导致更多的随机性和创造性，而较低的温度会导致更加确定性和保守性**")
st.sidebar.divider()

# 清理会话历史
buttonClean = st.sidebar.button("清理会话历史", key="clean", help="将会清空全部历史记录请慎重！！")
if buttonClean:
    st.session_state.history = []  # 清控历史对话记录
    st.session_state.past_key_values = None  # 清楚之前的键值
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空 CUDA 缓存。
    st.rerun()  # 重新运行应用程序，以便应用新的设置和状态更改。
st.sidebar.divider()
# 添加暂停按钮
pause = st.sidebar.button("暂停生成", help="暂停当前会话")

# 渲染聊天历史记录
for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    elif message["role"] == "observation":
        with st.chat_message(name="observation", avatar="assistant"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])


# 获取用户输入
prompt_text = st.chat_input("请输入您的问题")

# 如果用户输入了内容,则生成回复
if prompt_text:
    # 输入框和输出框
    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty()  # 创建一个空白容器，用于显示用户输入的消息和助手的回复消息
    input_placeholder.markdown(prompt_text)

    history = st.session_state.history
    past_key_values = st.session_state.past_key_values

    prompt_textBAC, Neo4j_data_str, Neo4j_data = entity_identification(prompt_text, word2id,
                                                                       id2label)

    # 识别出的实体不等于用户输入，证明识别出关键实体，可以进行知识图谱查询
    if Neo4j_data_str and prompt_textBAC != prompt_text:
    # if Neo4j_data_str:  # 测试
        # 查询Neo4j数据库，显示查询信息
        with st.chat_message(name="observation", avatar="assistant"):
            Neo4j_data_placeholder = st.empty()
        Neo4j_data_placeholder.markdown(Neo4j_data)
        # 修改对llm的提问
        prompt_textBAC = "对于问题：" + prompt_text + "。你可以借鉴下面信息：" + Neo4j_data_str + "。并添加更多的相关内容，请直接回答"

    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.empty()

    if not pause:
        for response, history, past_key_values in model.stream_chat(
                tokenizer,
                prompt_textBAC,
                history,
                past_key_values=past_key_values,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                return_past_key_values=True,
        ):
            message_placeholder.markdown(response)

            # 修改用户输入显示
            found_last_user_content = False
            for item in reversed(history):
                if item.get('role') == 'user' and not found_last_user_content:
                    if item.get('content') != prompt_text:
                        item['content'] = prompt_text
                    found_last_user_content = True

            #  添加知识图谱信息在历史中
            if Neo4j_data_str and prompt_textBAC != prompt_text:
            # if Neo4j_data_str:  # 测试
                st.session_state.history.append({"role": "observation", "content": Neo4j_data})

            # 跟新上下面信息
            st.session_state.history = history
            st.session_state.past_key_values = past_key_values
