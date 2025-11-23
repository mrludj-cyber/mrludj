import streamlit as st
import os
from google import genai
from google.genai import types
import time
import pandas as pd

# ==========================================
# [설정] API 키 & 프롬프트
# ==========================================
DEFAULT_API_KEY = "AIzaSyCre823lmqEE7re0lccGRqQMOkKLRoRQoI" 

SYSTEM_INSTRUCTION = """
당신은 유능한 **법률 전문가 AI**입니다. 
사용자가 업로드한 문서를 바탕으로 심도 있게 분석하여 답변하십시오.
반드시 *모든* 판례를 꼼꼼히 비교, 분석해서 *논거*로 삼고, *사례*를 제시해야 함.
*결론*제시보다 *논거, 사례* 제시가 중요함.

**[답변 스타일 가이드]**
1. **시각적 강조**: 핵심 법률 용어나 중요 문구는 파란색 배경으로 강조하십시오.
2. **결론 강조**: 결론 부분은 **굵게(Bold)** 표시하십시오.
3. **구조화**: ⚖️ **쟁점**, 🔍 **판단**, 💡 **결론** 등의 이모티콘을 사용하여 가독성을 높이십시오.
4. **참조 안내**: 판례 번호를 명시하고 원문 확인을 유도하십시오.
5. **참조 사례 리스트** : 질문과 관련하여 참고가 되는 판례번호를 모두 제시하십시오.
"""

st.set_page_config(page_title="Gemini Legal Search", page_icon="⚖️", layout="wide")

# ---------------------------------------------------------
# [CSS 수정] 강력한 강제 스타일링 적용
# ---------------------------------------------------------
st.markdown("""
<style>
    /* 1. 전체 앱 배경색 (연한 회색) */
    .stApp {
        background-color: #F8F9FA;
    }

    /* 2. 메인 컨테이너 하단 여백 확보 (입력창 가림 방지) */
    .main .block-container {
        padding-bottom: 120px !important;
    }

    /* [수정 2] 채팅 메시지 배경색 지정 */
    
    /* 사용자(User) 메시지: 파란색 배경 */
    [data-testid="stChatMessage"]:nth-of-type(odd) {
        background-color: #E3F2FD !important; 
        border: 1px solid #BBDEFB;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* AI(Assistant) 메시지: 흰색 배경 */
    [data-testid="stChatMessage"]:nth-of-type(even) {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }

    /* [수정 1] 입력창 스타일 (위치 강제 고정 CSS 제거 -> 기본 동작 활용) */
    /* 입력창 내부 디자인만 예쁘게 수정 */
    [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border-radius: 12px !important;
    }
    
    /* 입력창 테두리 및 그림자 */
    [data-testid="stChatInput"] > div {
        border-color: #BDBDBD !important; 
        border-radius: 12px !important;
        background-color: white !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    /* 입력창 포커스 시 */
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #1565C0 !important;
        box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.3) !important;
    }

    /* 팝업(Popover) 스타일 */
    [data-testid="stPopoverBody"] { border: 2px solid #2196F3; }
</style>
""", unsafe_allow_html=True)


# 세션 초기화
if "client" not in st.session_state: st.session_state.client = None
if "store" not in st.session_state: st.session_state.store = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ---------------------------------------------------------
# 기능 함수 (변경 없음)
# ---------------------------------------------------------
def initialize_client(api_key):
    try:
        os.environ["GEMINI_API_KEY"] = api_key
        client = genai.Client()
        return client, None
    except Exception as e:
        return None, str(e)

def create_store(client, store_name):
    try:
        store = client.file_search_stores.create(config={"display_name": store_name})
        return store, None
    except Exception as e:
        return None, str(e)

def get_all_stores(client):
    try:
        return list(client.file_search_stores.list()), None
    except Exception as e:
        return [], str(e)

def get_all_files_simple(client):
    try:
        all_files = list(client.files.list())
        file_data = []
        for f in all_files:
            size_bytes = getattr(f, 'size_bytes', 0)
            if size_bytes < 1024: size_str = f"{size_bytes} B"
            elif size_bytes < 1024**2: size_str = f"{size_bytes/1024:.1f} KB"
            else: size_str = f"{size_bytes/(1024**2):.1f} MB"

            file_data.append({
                "파일명": getattr(f, 'display_name', 'Unknown'),
                "상태": getattr(f, 'state', 'Unknown'),
                "크기": size_str,
                "생성일": str(getattr(f, 'create_time', 'Unknown'))[:10],
                "ID": f.name
            })
        return file_data
    except Exception as e:
        return []

def upload_file(client, file, store_name):
    try:
        import uuid
        file_ext = os.path.splitext(file.name)[1]
        temp_file = f"temp_{uuid.uuid4().hex}{file_ext}"
        with open(temp_file, "wb") as f:
            f.write(file.getbuffer())
        operation = client.file_search_stores.upload_to_file_search_store(
            file=temp_file,
            file_search_store_name=store_name,
            config={"display_name": file.name}
        )
        while not operation.done:
            time.sleep(1)
            try: operation = client.operations.get(operation)
            except: pass 
        if os.path.exists(temp_file): os.remove(temp_file)
        return True, None
    except Exception as e:
        return False, str(e)

def query_store_with_history(client, current_question, store_name, history):
    try:
        contents = []
        for chat in history:
            contents.append(types.Content(role="user", parts=[types.Part(text=chat["question"])]))
            contents.append(types.Content(role="model", parts=[types.Part(text=chat["answer"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=current_question)]))

        response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION, 
                tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name]))],
                temperature=0.1 
            )
        )
        citations = []
        if hasattr(response, "grounding_metadata") and response.grounding_metadata:
            if hasattr(response.grounding_metadata, "citations"):
                for citation in response.grounding_metadata.citations:
                    source_name = getattr(citation, "source", "문서")
                    if "/" in source_name: source_name = source_name.split("/")[-1]
                    citations.append({"source": source_name, "text": getattr(citation, "text", "")})
        return response.text, citations, None
    except Exception as e:
        return None, None, str(e)

# ---------------------------------------------------------
# UI 구성
# ---------------------------------------------------------
st.title("⚖️ Gemini Legal Search")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    api_key_input = st.text_input("API Key", value=DEFAULT_API_KEY if DEFAULT_API_KEY != "여기에_API_키를_입력하세요" else "", type="password")
    
    if api_key_input and not st.session_state.client:
        client, error = initialize_client(api_key_input)
        if client:
            st.session_state.client = client
            st.success("접속 성공")
            st.rerun()

    st.divider()
    
    if st.session_state.client:
        st.header("📁 Store 선택")
        stores, _ = get_all_stores(st.session_state.client)
        if stores:
            store_map = {s.display_name: s for s in stores}
            idx = 0
            if st.session_state.store and st.session_state.store.display_name in store_map:
                idx = list(store_map.keys()).index(st.session_state.store.display_name)
            selected = st.selectbox("사용할 Store", list(store_map.keys()), index=idx)
            if st.button("연결하기", use_container_width=True):
                st.session_state.store = store_map[selected]
                st.success(f"'{selected}' 연결됨")
                time.sleep(0.5)
                st.rerun()
        else:
            st.warning("Store가 없습니다.")
        with st.expander("새 Store 생성"):
            new_name = st.text_input("Store 이름")
            if st.button("생성"):
                s, e = create_store(st.session_state.client, new_name)
                if s: 
                    st.session_state.store = s
                    st.rerun()
        if st.session_state.store:
            st.info(f"**연결됨:** {st.session_state.store.display_name}")
        if st.button("🗑️ 대화 기록 지우기", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

if not st.session_state.client or not st.session_state.store:
    st.info("👈 왼쪽 사이드바에서 설정을 완료해주세요.")
    st.stop()

# 탭 구성
tab1, tab2 = st.tabs(["💬 법률 질의응답", "📂 파일 관리"])

# ---------------------------------------------------------
# Tab 1: 질의응답
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📘 문서 기반 법률 Q&A")

    # 대화 내용 표시
    for chat in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.write(chat["question"])
        
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(chat["answer"])
            if chat.get("citations"):
                st.markdown("---")
                st.markdown("**:blue[👇 참고 문헌 (판례 원문 보기)]**")
                cols = st.columns(min(3, len(chat["citations"]))) 
                for i, c in enumerate(chat["citations"]):
                    col_idx = i % 3
                    with cols[col_idx]:
                        short_source = c['source']
                        if len(short_source) > 12: short_source = short_source[:10] + "..."
                        with st.popover(f"📜 {short_source}", use_container_width=True):
                            st.markdown(f"### 📄 출처: {c['source']}")
                            st.divider()
                            st.info(c['text']) 

    # [입력창] 
    # CSS에서 stBottom을 position: fixed !important로 설정하여
    # 이 위젯이 어디에 선언되든 화면 최하단에 고정되도록 했습니다.
    if question := st.chat_input("판례나 법률 내용에 대해 질문하세요..."):
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("⚖️ 판례를 분석하고 있습니다..."):
                answer, citations, error = query_store_with_history(
                    st.session_state.client, question, st.session_state.store.name, st.session_state.chat_history
                )
                if answer:
                    st.markdown(answer)
                    if citations:
                        st.markdown("---")
                        st.markdown("**:blue[👇 참고 문헌 (판례 원문 보기)]**")
                        cols = st.columns(min(3, len(citations)))
                        for i, c in enumerate(citations):
                            col_idx = i % 3
                            with cols[col_idx]:
                                short_source = c['source']
                                if len(short_source) > 12: short_source = short_source[:10] + "..."
                                with st.popover(f"📜 {short_source}", use_container_width=True):
                                    st.markdown(f"### 📄 출처: {c['source']}")
                                    st.divider()
                                    st.info(c['text'])
                    st.session_state.chat_history.append({"question": question, "answer": answer, "citations": citations})
                else:
                    st.error(f"오류가 발생했습니다: {error}")

# ---------------------------------------------------------
# Tab 2: 파일 관리
# ---------------------------------------------------------
with tab2:
    st.header("📂 전체 파일 목록")
    if st.button("🔄 새로고침"): st.rerun()
    file_data = get_all_files_simple(st.session_state.client)
    if file_data:
        df = pd.DataFrame(file_data)
        st.dataframe(df[["파일명", "크기", "상태", "생성일", "ID"]], use_container_width=True, hide_index=True)
    else:
        st.info("조회된 파일이 없습니다.")
    st.divider()
    st.subheader("새 파일 업로드")
    uploaded = st.file_uploader("파일 선택", accept_multiple_files=True)
    if uploaded and st.button("업로드 시작"):
        progress = st.progress(0)
        for i, f in enumerate(uploaded):
            upload_file(st.session_state.client, f, st.session_state.store.name)
            progress.progress((i+1)/len(uploaded))
        st.success("완료! 목록을 갱신합니다.")
        time.sleep(1)
        st.rerun()
