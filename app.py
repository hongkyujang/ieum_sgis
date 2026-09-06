import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 기본 설정 및 디자인
st.set_page_config(
    page_title="SIGS 대학생-강소기업 듀얼 매칭 플랫폼", 
    page_icon="💼",
    layout="wide"
)

# 2. 세션 상태(Session State)를 활용한 동적 데이터베이스 초기화
if "jobs_db" not in st.session_state:
    st.session_state.jobs_db = pd.DataFrame([
        {
            "job_id": "J01",
            "company_name": "(주)알파인텔리전스",
            "company_type": "강소기업",
            "target_major": "컴퓨터공학과",
            "sig_category": "AI/ML",
            "required_skills": "Python PyTorch 머신러닝 딥러닝 OpenCV",
            "location": "서울 강남구",
            "salary": "3,800만원"
        },
        {
            "job_id": "J02",
            "company_name": "(주)베타시스템즈",
            "company_type": "중견기업",
            "target_major": "전자공학과",
            "sig_category": "임베디드",
            "required_skills": "C++ 임베디드 펌웨어 회로설계 RTOS",
            "location": "경기 성남시",
            "salary": "4,200만원"
        },
        {
            "job_id": "J03",
            "company_name": "(주)감마데이터",
            "company_type": "중소기업",
            "target_major": "컴퓨터공학과",
            "sig_category": "데이터 엔지니어링",
            "required_skills": "Python SQL 데이터엔지니어링 ETL Spark",
            "location": "서울 금천구",
            "salary": "3,600만원"
        }
    ])

if "students_db" not in st.session_state:
    st.session_state.students_db = pd.DataFrame([
        {
            "student_id": "S01",
            "name": "김철수",
            "major": "컴퓨터공학과",
            "sig_interests": "AI/ML, 데이터 엔지니어링",
            "skills": "Python PyTorch 데이터분석 머신러닝 OpenCV",
            "email": "chulsoo@university.ac.kr"
        },
        {
            "student_id": "S02",
            "name": "이영희",
            "major": "전자공학과",
            "sig_interests": "임베디드",
            "skills": "C++ 임베디드 펌웨어 회로설계 RTOS",
            "email": "younghee@university.ac.kr"
        },
        {
            "student_id": "S03",
            "name": "박민수",
            "major": "컴퓨터공학과",
            "sig_interests": "데이터 엔지니어링",
            "skills": "Python SQL 데이터분석 ETL Spark",
            "email": "minsu@university.ac.kr"
        }
    ])

# 3. 메인 타이틀 및 서비스 모드 선택
st.title("🤝 SIGS 대학생 ↔ 강소/중견기업 상호 정밀 매칭 플랫폼")
st.caption("대학생은 전공에 맞는 숨은 알짜 기업을, 기업은 요구 직무 스킬에 딱 맞는 전공 인재를 찾습니다.")

# 사이드바 모드 전환
mode = st.sidebar.radio(
    "📌 서비스 모드 선택",
    ["🎓 대학생 (맞춤 기업 찾기)", "🏢 기업 담당자 (공고 등록 & 인재 검색)"]
)

st.sidebar.markdown("---")

# ==========================================
# 모드 1: 대학생 모드
# ==========================================
if mode == "🎓 대학생 (맞춤 기업 찾기)":
    st.header("🎓 대학생 맞춤형 강소·중견기업 매칭")
    
    st.sidebar.subheader("👤 대학생 프로필 등록/매칭")
    student_name = st.sidebar.text_input("이름", value="김철수")
    student_major = st.sidebar.selectbox(
        "전공 선택", 
        ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과"]
    )
    student_interests = st.sidebar.multiselect(
        "SIGS 관심 분야 선택",
        ["AI/ML", "데이터 엔지니어링", "임베디드", "바이오/제약", "클라우드"],
        default=["AI/ML", "데이터 엔지니어링"]
    )
    student_skills = st.sidebar.text_area(
        "보유 기술 키워드 (공백 구분)",
        value="Python PyTorch 데이터분석 머신러닝"
    )

    if st.sidebar.button("🔍 맞춤 기업 매칭 실행"):
        jobs_df = st.session_state.jobs_db
        if jobs_df.empty:
            st.warning("현재 등록된 기업 채용 공고가 없습니다.")
        elif not student_skills.strip():
            st.warning("보유 기술 키워드를 입력해주세요.")
        else:
            # TF-IDF 스킬 유사도 계산
            all_skills = list(jobs_df['required_skills']) + [student_skills]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_skills)
            
            student_vec = tfidf_matrix[-1]
            job_vecs = tfidf_matrix[:-1]
            skill_sims = cosine_similarity(student_vec, job_vecs).flatten()

            # SIGS 매칭 점수 산출
            results = []
            for idx, row in jobs_df.iterrows():
                skill_score = skill_sims[idx] * 100
                major_score = 20 if row['target_major'] == student_major else 0
                sig_score = 30 if any(sig in row['sig_category'] for sig in student_interests) else 0
                
                final_score = (skill_score * 0.5) + major_score + sig_score
                
                results.append({
                    "매칭 점수": round(final_score, 1),
                    "회사명": row['company_name'],
                    "기업 유형": row['company_type'],
                    "SIG 분야": row['sig_category'],
                    "타겟 전공": row['target_major'],
                    "요구 스킬": row['required_skills'],
                    "위치": row['location'],
                    "연봉": row['salary']
                })

            res_df = pd.DataFrame(results).sort_values(by="매칭 점수", ascending=False)
            
            top_job = res_df.iloc[0]
            st.success(f"🔥 **{student_name}**님께 가장 추천하는 기업: **{top_job['회사명']}** (매칭률 **{top_job['매칭 점수']}점**)")
            
            st.dataframe(
                res_df,
                column_config={"매칭 점수": st.column_config.NumberColumn(format="%d점")},
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("👈 좌측 사이드바에서 프로필을 설정하고 '맞춤 기업 매칭 실행'을 눌러주세요.")
        st.subheader("📋 전체 채용 공고 목록")
        st.dataframe(st.session_state.jobs_db, use_container_width=True, hide_index=True)

# ==========================================
# 모드 2: 기업 담당자 모드
# ==========================================
else:
    st.header("🏢 기업 전용 서비스: 신규 공고 등록 & 인재 매칭")
    
    tab1, tab2 = st.tabs(["➕ 신규 채용 공고 등록", "🔎 맞춤형 전공 인재 발굴"])
    
    # 탭 1: 기업의 채용 공고 등록
    with tab1:
        st.subheader("📝 기업 채용 정보 입력")
        with st.form("job_post_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                company_name = st.text_input("기업명 (예: (주)테크솔루션)")
                company_type = st.selectbox("기업 구분", ["강소기업", "중견기업", "중소기업", "스타트업"])
                target_major = st.selectbox("우선 선호 전공", ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과"])
                location = st.text_input("근무지 (예: 서울 강남구)")
            with col2:
                sig_category = st.selectbox("SIGS 직무 분야", ["AI/ML", "데이터 엔지니어링", "임베디드", "바이오/제약", "클라우드"])
                required_skills = st.text_input("필요 역량/스킬 키워드 (공백 구분)", value="Python SQL C++")
                salary = st.text_input("연봉 조건 (예: 3,800만원)")
            
            submit_btn = st.form_submit_button("📢 채용 공고 즉시 등록")
            
            if submit_btn:
                if not company_name or not required_skills:
                    st.error("기업명과 필요 역량 키워드는 필수 항목입니다.")
                else:
                    new_job = {
                        "job_id": f"J{len(st.session_state.jobs_db) + 1:02d}",
                        "company_name": company_name,
                        "company_type": company_type,
                        "target_major": target_major,
                        "sig_category": sig_category,
                        "required_skills": required_skills,
                        "location": location,
                        "salary": salary
                    }
                    st.session_state.jobs_db = pd.concat(
                        [st.session_state.jobs_db, pd.DataFrame([new_job])], 
                        ignore_index=True
                    )
                    st.success(f"✅ '{company_name}'의 공고가 성공적으로 등록되었습니다! 학생 추천 DB에 즉시 반영됩니다.")

    # 탭 2: 기업이 인재(대학생 DB)를 스킬 기반으로 매칭
    with tab2:
        st.subheader("🎯 기업 조건 맞춤 인재 탐색")
        
        selected_job_id = st.selectbox(
            "인재를 검색할 자사의 채용 공고 선택",
            options=st.session_state.jobs_db["job_id"].tolist(),
            format_func=lambda x: f"[{x}] {st.session_state.jobs_db[st.session_state.jobs_db['job_id']==x]['company_name'].values[0]} ({st.session_state.jobs_db[st.session_state.jobs_db['job_id']==x]['sig_category'].values[0]})"
        )
        
        if st.button("🔎 해당 공고에 적합한 인재 검색"):
            selected_job = st.session_state.jobs_db[st.session_state.jobs_db["job_id"] == selected_job_id].iloc[0]
            students_df = st.session_state.students_db
            
            if students_df.empty:
                st.warning("등록된 대학생 인재 데이터가 없습니다.")
            else:
                # 기업 필요 스킬과 대학생 보유 스킬 간 TF-IDF 연산
                all_skills = [selected_job['required_skills']] + list(students_df['skills'])
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(all_skills)
                
                job_vec = tfidf_matrix[0]
                student_vecs = tfidf_matrix[1:]
                skill_sims = cosine_similarity(job_vec, student_vecs).flatten()

                match_results = []
                for idx, student in students_df.iterrows():
                    skill_score = skill_sims[idx] * 100
                    major_score = 20 if student['major'] == selected_job['target_major'] else 0
                    sig_score = 30 if selected_job['sig_category'] in student['sig_interests'] else 0
                    
                    final_score = (skill_score * 0.5) + major_score + sig_score
                    
                    match_results.append({
                        "적합도 점수": round(final_score, 1),
                        "이름": student['name'],
                        "전공": student['major'],
                        "SIGS 관심사": student['sig_interests'],
                        "보유 스킬": student['skills'],
                        "연락처(이메일)": student['email']
                    })

                res_students = pd.DataFrame(match_results).sort_values(by="적합도 점수", ascending=False)
                
                st.write(f"### 💡 **[{selected_job['company_name']}]** 맞춤 추천 인재 리스트")
                st.dataframe(
                    res_students,
                    column_config={"적합도 점수": st.column_config.NumberColumn(format="%d점")},
                    use_container_width=True,
                    hide_index=True
                )
