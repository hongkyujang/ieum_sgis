import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 기본 설정 및 브랜딩
st.set_page_config(
    page_title="SGIS 공간위치 기반 대학생-강소기업 매칭 플랫폼", 
    page_icon="🗺️",
    layout="wide"
)

# 2. 세션 상태(Session State) 데이터베이스 초기화
if "jobs_db" not in st.session_state:
    st.session_state.jobs_db = pd.DataFrame([
        {
            "job_id": "J01",
            "company_name": "루닛 (Lunit)",
            "company_type": "강소/코스닥상장",
            "target_major": "컴퓨터공학과",
            "sig_category": "AI/ML",
            "required_skills": "Python PyTorch 딥러닝 의료영상분석 ComputerVision OpenCV",
            "address": "서울 강남구 테헤란로 211",
            "lat": 37.5032,
            "lon": 127.0416,
            "salary": "4,500만원"
        },
        {
            "job_id": "J02",
            "company_name": "센드버드 (Sendbird)",
            "company_type": "유니콘/강소기업",
            "target_major": "컴퓨터공학과",
            "sig_category": "클라우드",
            "required_skills": "Python Go AWS Docker Kubernetes DistributedSystems API",
            "address": "서울 강남구 테헤란로 142",
            "lat": 37.5002,
            "lon": 127.0365,
            "salary": "4,800만원"
        },
        {
            "job_id": "J03",
            "company_name": "원티드랩 (Wanted Lab)",
            "company_type": "중견/코스닥상장",
            "target_major": "컴퓨터공학과",
            "sig_category": "데이터 엔지니어링",
            "required_skills": "Python SQL 데이터엔지니어링 Spark ETL BigQuery Pandas",
            "address": "서울 송파구 올림픽로 300",
            "lat": 37.5137,
            "lon": 127.1042,
            "salary": "4,200만원"
        },
        {
            "job_id": "J04",
            "company_name": "리벨리온 (Rebellions)",
            "company_type": "강소/스타트업",
            "target_major": "전자공학과",
            "sig_category": "임베디드",
            "required_skills": "C++ C 임베디드 FPGA SoC 회로설계 반도체설계 RTL",
            "address": "경기 성남시 분당구 판교역로 166",
            "lat": 37.3952,
            "lon": 127.1114,
            "salary": "4,600만원"
        },
        {
            "job_id": "J05",
            "company_name": "알테오젠 (AlteoGen)",
            "company_type": "중견기업",
            "target_major": "생명공학과",
            "sig_category": "바이오/제약",
            "required_skills": "단백질공학 유전자재조합 PCR 바이오의약품 HPLC 분석화학",
            "address": "대전 유성구 문지로 281-25",
            "lat": 36.3980,
            "lon": 127.3995,
            "salary": "4,000만원"
        },
        {
            "job_id": "J06",
            "company_name": "두산로보틱스 (Doosan Robotics)",
            "company_type": "중견/대기업 계열",
            "target_major": "기계공학과",
            "sig_category": "로봇/제어",
            "required_skills": "C++ ROS 로봇제어 역학설계 CAD 3D모데링 자동화",
            "address": "경기 수원시 영통구 삼성로 156",
            "lat": 37.2636,
            "lon": 127.0514,
            "salary": "4,500만원"
        }
    ])

if "students_db" not in st.session_state:
    st.session_state.students_db = pd.DataFrame([
        {
            "student_id": "S01",
            "name": "김철수",
            "major": "컴퓨터공학과",
            "sig_interests": "AI/ML, 데이터 엔지니어링",
            "skills": "Python PyTorch 딥러닝 ComputerVision OpenCV 데이터분석",
            "email": "chulsoo.kim@univ.ac.kr"
        }
    ])

# 지오코더(주소 -> 위경도 변환 함수)
@st.cache_data
def geocode_address(address_str):
    try:
        geolocator = Nominatim(user_agent="sigs_job_platform")
        location = geolocator.geocode(address_str)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    return None, None

# 3. 메인 타이틀
st.title("🗺️ SGIS 공간위치 기반 대학생 ↔ 강소기업 정밀 매칭 플랫폼")
st.caption("내 주소를 직접 입력하여 위치 기반으로 근거리 알짜 기업을 지도 위 핀으로 조회하고 역량 매칭을 진행합니다.")

# 사이드바 모드 전환
mode = st.sidebar.radio(
    "📌 서비스 모드 선택",
    ["🎓 대학생 (내 근처 맞춤 기업 지도 탐색)", "🏢 기업 담당자 (공고 등록 & 인재 검색)"]
)

st.sidebar.markdown("---")

# ==========================================
# 모드 1: 대학생 모드 (사용자 주소 직접 입력 기능 연동)
# ==========================================
if mode == "🎓 대학생 (내 근처 맞춤 기업 지도 탐색)":
    st.header("🎓 SGIS 위치 기반 내 근처 기업 탐색 및 직무 매칭")
    
    st.sidebar.subheader("👤 대학생 프로필 & 직접 위치 입력")
    student_name = st.sidebar.text_input("이름", value="김철수")
    student_major = st.sidebar.selectbox(
        "전공 선택", 
        ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과"]
    )
    
    # 주소 직접 입력창 기능
    input_address = st.sidebar.text_input(
        "📍 내 현재 주소/거주지 직접 입력",
        value="서울시 강남구 역삼동"
    )
    
    # 입력한 주소를 위/경도 좌표로 변환
    user_lat, user_lon = geocode_address(input_address)
    
    # 주소 검색 실패 시 기본 좌표(서울 강남) 예외 처리
    if user_lat is None or user_lon is None:
        st.sidebar.warning("⚠️ 주소를 찾을 수 없어 기본 위치(서울 강남)로 설정됩니다.")
        user_lat, user_lon = 37.4979, 127.0276
    else:
        st.sidebar.success(f"📍 위치 인식 완료: ({round(user_lat, 4)}, {round(user_lon, 4)})")

    max_distance_km = st.sidebar.slider("📏 통근 가능 최대 반경 범위 (km)", min_value=1, max_value=50, value=15)
    
    student_interests = st.sidebar.multiselect(
        "SIGS 관심 분야 선택",
        ["AI/ML", "데이터 엔지니어링", "임베디드", "바이오/제약", "클라우드", "로봇/제어"],
        default=["AI/ML", "데이터 엔지니어링"]
    )
    student_skills = st.sidebar.text_area(
        "보유 기술/역량 키워드 (공백 구분)",
        value="Python PyTorch 딥러닝 ComputerVision OpenCV 데이터분석"
    )

    # 거리 연산 및 공간 필터링
    jobs_df = st.session_state.jobs_db.copy()
    
    def calc_dist(row):
        return round(geodesic((user_lat, user_lon), (row['lat'], row['lon'])).km, 2)

    jobs_df['distance_km'] = jobs_df.apply(calc_dist, axis=1)
    filtered_jobs = jobs_df[jobs_df['distance_km'] <= max_distance_km].copy()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader(f"📍 입력 위치('{input_address}') 기반 지도 시각화")
        
        # 지도 생성 (입력받은 위경도 중심으로 세팅)
        m = folium.Map(location=[user_lat, user_lon], zoom_start=11)
        
        # 입력한 사용자 위치 핀
        folium.Marker(
            location=[user_lat, user_lon],
            popup=f"<b>[내 위치] {student_name}님</b><br>{input_address}",
            tooltip=f"내 위치 ({input_address})",
            icon=folium.Icon(color="red", icon="user", prefix="fa")
        ).add_to(m)

        # 설정 반경 원 표시
        folium.Circle(
            location=[user_lat, user_lon],
            radius=max_distance_km * 1000,
            color="#3186cc",
            fill=True,
            fill_opacity=0.1
        ).add_to(m)

        # 반경 내 기업 핀 표시
        for _, row in filtered_jobs.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"<b>{row['company_name']}</b><br>{row['sig_category']}<br>주소: {row['address']}<br>거리: {row['distance_km']}km",
                tooltip=f"{row['company_name']} ({row['distance_km']}km)",
                icon=folium.Icon(color="blue", icon="building", prefix="fa")
            ).add_to(m)

        st_folium(m, width="100%", height=450)

    with col2:
        st.subheader(f"🎯 반경 {max_distance_km}km 내 추천 기업 리스트")
        
        if filtered_jobs.empty:
            st.warning(f"입력하신 위치에서 반경 {max_distance_km}km 내에 등록된 기업이 없습니다.")
        else:
            all_skills = list(filtered_jobs['required_skills']) + [student_skills]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_skills)
            
            student_vec = tfidf_matrix[-1]
            job_vecs = tfidf_matrix[:-1]
            skill_sims = cosine_similarity(student_vec, job_vecs).flatten()

            results = []
            for idx, (_, row) in enumerate(filtered_jobs.iterrows()):
                skill_score = skill_sims[idx] * 100
                major_score = 20 if row['target_major'] == student_major else 0
                sig_score = 30 if any(sig in row['sig_category'] for sig in student_interests) else 0
                
                final_score = (skill_score * 0.5) + major_score + sig_score
                
                results.append({
                    "매칭 점수": round(final_score, 1),
                    "기업명": row['company_name'],
                    "거리(km)": f"{row['distance_km']} km",
                    "SIG 분야": row['sig_category'],
                    "요구 스킬": row['required_skills'],
                    "연봉": row['salary']
                })

            res_df = pd.DataFrame(results).sort_values(by="매칭 점수", ascending=False)
            
            top_company = res_df.iloc[0]
            st.success(f"🔥 추천 1순위: **{top_company['기업명']}** (거리: {top_company['거리(km)']}, 매칭률: **{top_company['매칭 점수']}점**)")
            
            st.dataframe(
                res_df,
                column_config={"매칭 점수": st.column_config.NumberColumn(format="%d점")},
                use_container_width=True,
                hide_index=True
            )

# ==========================================
# 모드 2: 기업 담당자 모드
# ==========================================
else:
    st.header("🏢 기업 전용 서비스: 공고 등록 및 SGIS 인재 탐색")
    
    tab1, tab2 = st.tabs(["➕ 신규 채용 공고 등록", "🔎 맞춤형 전공 인재 발굴"])
    
    with tab1:
        st.subheader("📝 신규 공고 작성 (주소 입력 시 좌표 자동 변환)")
        with st.form("job_post_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                company_name = st.text_input("기업명")
                company_type = st.selectbox("기업 구분", ["강소기업", "중견기업", "유니콘/강소", "스타트업"])
                target_major = st.selectbox("우선 선호 전공", ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과"])
                address = st.text_input("기업 도로명 주소", value="서울시 마포구 상암동")
            with col2:
                sig_category = st.selectbox("SIGS 직무 분야", ["AI/ML", "데이터 엔지니어링", "임베디드", "바이오/제약", "클라우드", "로봇/제어"])
                required_skills = st.text_input("필요 역량/스킬 키워드 (공백 구분)", value="Python C++ 딥러닝")
                salary = st.text_input("연봉 조건", value="4,000만원")
            
            submit_btn = st.form_submit_button("📢 SGIS 지도 등록 공고 제출")
            
            if submit_btn:
                if not company_name or not required_skills:
                    st.error("기업명과 필요 역량 키워드는 필수 사항입니다.")
                else:
                    lat, lon = geocode_address(address)
                    if lat is None or lon is None:
                        lat, lon = 37.5665, 126.9780  # 변환 실패 시 기본 좌표
                    
                    new_job = {
                        "job_id": f"J{len(st.session_state.jobs_db) + 1:02d}",
                        "company_name": company_name,
                        "company_type": company_type,
                        "target_major": target_major,
                        "sig_category": sig_category,
                        "required_skills": required_skills,
                        "address": address,
                        "lat": lat,
                        "lon": lon,
                        "salary": salary
                    }
                    st.session_state.jobs_db = pd.concat(
                        [st.session_state.jobs_db, pd.DataFrame([new_job])], 
                        ignore_index=True
                    )
                    st.success(f"✅ [{company_name}] 공고가 좌표({round(lat,4)}, {round(lon,4)}) 변환되어 지도에 성공적으로 등록되었습니다!")

    with tab2:
        st.subheader("🎯 기업 맞춤형 등록 인재 조회")
        st.dataframe(st.session_state.students_db, use_container_width=True, hide_index=True)
