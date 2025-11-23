import streamlit as st
import os
from google import genai
from google.genai import types
import time
import pandas as pd

# ==========================================
# [설정] API 키 & 프롬프트
# ==========================================
DEFAULT_API_KEY = "" 

SYSTEM_INSTRUCTION = """
당신은 유능한 법률 전문가 팀(리서치 담당관 + 수석 변호사)입니다.
사용자의 질문에 답하기 위해 반드시 **다음 2단계 프로세스**를 거쳐 답변을 작성해야 합니다.

**[Step 1: 리서치 단계 (Fact Finding)]**
* 먼저 문서 저장소(Store)를 검색하여 질문과 관련된 **모든 판례, 법령, 사실관계**를 있는 그대로 발췌하십시오.
* 이 단계에서는 주관적인 의견을 배제하고, **"어떤 판례가 발견되었는가?"**에만 집중하십시오.
* 발견된 판례는 `판례번호`, `판시사항`, `판결요지` 핵심을 요약하여 리스트화하십시오.
* 사용자의 질문과 관련된 판례가 문서 내에 존재한다면, **중요도나 대표성을 따지지 말고 발견되는 모든 판례를 나열**해야 합니다.
*[Step 1의 핵심 행동 수칙]*
1. **Selection(선별) 금지**: "대표적인 판례 몇 가지만 소개합니다"라는 태도를 버리십시오. 비슷한 판례라도 사건 번호가 다르면 모두 나열하십시오.
2. **Exhaustive Listing(포괄적 나열)**: 검색된 컨텍스트(Context) 내에 있는 판례가 10개면 10개, 20개면 20개를 전부 표에 적으십시오.
3. **그룹화(Grouping)**: 판례가 너무 많을 경우, [유사 쟁점] 또는 [연도별]로 그룹화하여 가독성을 높이되, 절대 생략하지 마십시오.

**[Step 2: 종합 분석 단계 (Legal Analysis)]**
* [Step 1]에서 찾은 재료들을 바탕으로 심층적인 법률 검토 의견서를 작성하십시오.
* 단순한 나열이 아니라, 수집된 판례들이 **이 사안에서 어떻게 해석되는지, 상충되는 견해는 없는지, 최종적으로 어떤 결론에 도달하는지** 논리적으로 서술하십시오.
* 결론을 도출할 때는 반드시 [Step 1]의 판례 번호를 인용(Citation)하여 근거를 대십시오.

**[출력 형식 가이드]**
답변은 반드시 아래 두 섹션으로 구분되어야 합니다.

---
### 1. 🗂️ 기초 조사 보고 (관련 판례 및 법령)
**[답변 형식]**

**1. 🗂️ 발견된 전체 판례 리스트 (Total List)**
* "문서 검색 결과, 총 N건의 관련 판례가 발견되었습니다."라고 시작할 것.
* 아래 표 형식으로 최대한 길게 작성할 것.
| 연번 | 사건번호 | 판결일자 | 핵심 요지 (3줄 요약) |
| :-- | :--- | :--- | :--- |
| 1 | 2023다12345 | 2023.01.01 | 임대차 보증금 반환 의무 |
| 2 | 2022다56789 | 2022.05.20 | 위와 유사 취지의 판결 |
... (발견된 모든 건수 기재)

**2. 📜 주요 판례 상세 분석**
* 위 리스트 중 가장 쟁점과 부합하는 핵심 판례 10개를 선정하여 상세 내용을 서술.

**3. ⚖️ 종합 결론**
* 다수 판례의 경향성(Trend) 분석.

### 2. ⚖️ 종합 법률 검토 의견
(여기에 위 조사 내용을 종합한 전문가의 심층 분석 및 결론 제시)
---
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
        # [수정] 조사 -> 종합 순서로 사고하도록 유도하는 프롬프트
        enhanced_question = f"""
        [사용자 질문]: {current_question}
        
        [수행 지침]:
        1. 먼저 업로드된 문서들 중에서 위 질문과 연관된 판례, 법 조항, 핵심 문구를 **빠짐없이 검색(Search)**하여 나열하십시오.
        2. 제공된 문서 컨텍스트(Context)를 바닥까지 긁어서(Exhaustively Search), 질문과 조금이라도 관련된 판례는 **단 하나도 빠뜨리지 말고 모두 나열**하십시오.
        3. 만약 판례가 10개 이상 발견되면 10개 이상 모두 적으십시오. "그 외 다수 있음"이라고 줄이지 마십시오.
        4. 단순한 요약보다는 **최대한 많은 판례 번호(Case Number)**를 확보하는 것이 이번 작업의 목표입니다.
        5. 동일한 법리를 다루더라도 사건 번호가 다르면 별개의 항목으로 취급하여 리스트에 포함시키십시오.
        6. 그 다음, 검색된 자료들을 **종합(Synthesize)**하여 논리적인 법률 검토 의견을 서술하십시오.
        7. 조사가 부실하면 분석도 부실해집니다. **최대한 많은 근거**를 확보한 뒤 분석을 시작하십시오.
        """

        contents = []
        for chat in history:
            contents.append(types.Content(role="user", parts=[types.Part(text=chat["question"])]))
            contents.append(types.Content(role="model", parts=[types.Part(text=chat["answer"])]))
        
        contents.append(types.Content(role="user", parts=[types.Part(text=enhanced_question)]))

        response = client.models.generate_content(
            model="gemini-3-pro-preview", # [필수] 복합 추론을 위해 고성능 모델 사용
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name]))],
                temperature=0.1,        # [설정] 0.1 : 사실에 기반하되 약간의 문장 구성력 허용
                max_output_tokens=8192 
            )
        )
        
        # (Citations 처리 코드는 기존과 동일)
        citations = []
        if hasattr(response, "grounding_metadata") and response.grounding_metadata:
            if hasattr(response.grounding_metadata, "citations"):
                for citation in response.grounding_metadata.citations:
                    source_name = getattr(citation, "source", None)
                    if not source_name: source_name = getattr(citation, "title", "문서")
                    if isinstance(source_name, str) and "/" in source_name: 
                        source_name = source_name.split("/")[-1]
                    
                    citations.append({
                        "source": source_name, 
                        "text": getattr(citation, "text", "")
                    })

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
