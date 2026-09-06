import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 페이지 기본 설정
st.set_page_config(page_title="SIGS 대학생-강소기업 매칭 플랫폼", layout="wide")

# Mock 데이터베이스 (기업 채용 공고)
@st.cache_data
def load_jobs_data():
    return pd.DataFrame([
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
        },
        {
            "job_id": "J04",
            "company_name": "(주)델타바이오",
            "company_type": "강소기업",
            "target_major": "생명공학과",
            "sig_category": "바이오/제약",
            "required_skills": "유전자분석 PCR 바이오인포매틱스 Python",
            "location": "대전 유성구",
            "salary": "3,500만원"
        }
    ])

# UI 타이틀
st.title("🎓 SIGS 전공 맞춤형 강소기업 채용 플랫폼")
st.caption("잘 알려지지 않은 알짜 중소·중견기업과 전공 인재를 SIGS 기반으로 정밀 매칭합니다.")

jobs_df = load_jobs_data()

# 사이드바: 대학생 인재 정보 입력 폼
st.sidebar.header("👤 대학생 프로필 등록")
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
    "보유 기술 키워드 (공백으로 구분)",
    value="Python PyTorch 데이터분석 머신러닝"
)

# 메인 화면: 매칭 엔진 연산 및 결과 제공
if st.sidebar.button("맞춤형 기업 매칭 실행"):
    if not student_skills.strip():
        st.warning("보유 기술 키워드를 입력해주세요.")
    else:
        # TF-IDF 스킬 유사도 계산
        all_skills = list(jobs_df['required_skills']) + [student_skills]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_skills)
        
        student_vec = tfidf_matrix[-1]
        job_vecs = tfidf_matrix[:-1]
        
        skill_sims = cosine_similarity(student_vec, job_vecs).flatten()

        # SIGS 점수 계산 (스킬 50% + 전공 20% + SIGS 관심사 30%)
        results = []
        for idx, row in jobs_df.iterrows():
            skill_score = skill_sims[idx] * 100
            major_score = 20 if row['target_major'] == student_major else 0
            sig_score = 30 if row['sig_category'] in student_interests else 0
            
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

        st.subheader(f"🎯 {student_name}님을 위한 SIGS 추천 기업 리스트")
        
        # 카드 형태로 상위 기업 강조 출력
        top_job = res_df.iloc[0]
        st.success(f"🔥 **최고 매칭 기업**: {top_job['회사명']} ({top_job['기업 유형']}) - 매칭률 **{top_job['매칭 점수']}점**")

        # 결과 데이터 프레임 출력 (잡코리아 스타일)
        st.dataframe(
            res_df,
            column_config={
                "매칭 점수": st.column_config.NumberColumn(format="%d점"),
            },
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("👈 좌측 사이드바에서 프로필과 보유 기술을 입력한 뒤 '맞춤형 기업 매칭 실행' 버튼을 눌러주세요.")
    
    st.subheader("📋 전체 등록된 숨은 강소/중견기업 공고")
    st.dataframe(jobs_df, use_container_width=True, hide_index=True)