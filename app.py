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

# 2. 세션 상태(Session State) 데이터베이스 초기화 (다양한 전공 및 권역별 DB 구축)
if "jobs_db" not in st.session_state:
    st.session_state.jobs_db = pd.DataFrame([
        # --- [서울 강남/성수/서초 권역 - IT, 경영, 디자인, 데이터] ---
        {"job_id": "J01", "company_name": "루닛 (Lunit)", "company_type": "강소/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python PyTorch 딥러닝 의료영상분석 ComputerVision OpenCV", "address": "서울 강남구 테헤란로 211", "lat": 37.5032, "lon": 127.0416, "salary": "4,500만원"},
        {"job_id": "J02", "company_name": "센드버드 (Sendbird)", "company_type": "유니콘/강소", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Python Go AWS Docker Kubernetes API 백엔드", "address": "서울 강남구 테헤란로 142", "lat": 37.5002, "lon": 127.0365, "salary": "4,800만원"},
        {"job_id": "J03", "company_name": "원티드랩 (Wanted Lab)", "company_type": "중견/상장", "target_major": "경영학과", "sig_category": "마케팅/기획", "required_skills": "퍼포먼스마케팅 데이터분석 SQL 서비스기획 마케팅전략 GA4", "address": "서울 송파구 올림픽로 300", "lat": 37.5137, "lon": 127.1042, "salary": "4,000만원"},
        {"job_id": "J04", "company_name": "몰로코 (Moloco)", "company_type": "유니콘/강소", "target_major": "산업공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python SQL 머신러닝 최적화 데이터분석 통계분석 빅데이터", "address": "서울 강남구 테헤란로 521", "lat": 37.5088, "lon": 127.0608, "salary": "5,000만원"},
        {"job_id": "J05", "company_name": "뤼이드 (Riiid)", "company_type": "강소기업", "target_major": "디자인학과", "sig_category": "UI/UX 디자인", "required_skills": "Figma UIUX디자인 프로토타이핑 Figma사용 유저리서치 서비스디자인", "address": "서울 강남구 영동대로 517", "lat": 37.5126, "lon": 127.0588, "salary": "4,200만원"},
        {"job_id": "J06", "company_name": "쏘카 (SOCAR)", "company_type": "중견/상장", "target_major": "경영학과", "sig_category": "마케팅/기획", "required_skills": "사업기획 데이터기반의사결정 프로덕트기획 SQL KPI관리 분석", "address": "서울 성동구 아차산로 6", "lat": 37.5467, "lon": 127.0435, "salary": "4,300만원"},
        {"job_id": "J07", "company_name": "직방 (Zigbang)", "company_type": "강소기업", "target_major": "디자인학과", "sig_category": "UI/UX 디자인", "required_skills": "Figma Sketch UIUX 웹디자인 브랜드디자인 그래픽디자인", "address": "서울 서초구 서초대로 398", "lat": 37.4975, "lon": 127.0253, "salary": "4,000만원"},
        {"job_id": "J08", "company_name": "버킷플레이스 (오늘의집)", "company_type": "유니콘/강소", "target_major": "경영학과", "sig_category": "마케팅/기획", "required_skills": "브랜드마케팅 콘텐츠기획 이커머스기획 데이터분석 마케팅", "address": "서울 서초구 서초대로 301", "lat": 37.4988, "lon": 127.0160, "salary": "4,500만원"},
        {"job_id": "J09", "company_name": "당근 (Daangn)", "company_type": "유니콘/강소", "target_major": "산업공학과", "sig_category": "데이터 엔지니어링", "required_skills": "SQL Python 데이터기반기획 프로세스최적화 지표분석 AB테스트", "address": "서울 서초구 강남대로 373", "lat": 37.4939, "lon": 127.0290, "salary": "4,600만원"},
        {"job_id": "J10", "company_name": "야놀자 (Yanolja)", "company_type": "중견/유니콘", "target_major": "경영학과", "sig_category": "마케팅/기획", "required_skills": "글로벌마케팅 전략기획 파트너십 이커머스 사업개발", "address": "서울 강남구 테헤란로 108", "lat": 37.4988, "lon": 127.0289, "salary": "4,400만원"},

        # --- [경기 판교/분당 IT·제조·공학 밸리] ---
        {"job_id": "J11", "company_name": "리벨리온 (Rebellions)", "company_type": "강소/스타트업", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "C++ C 임베디드 FPGA SoC 회로설계 반도체설계 RTL", "address": "경기 성남시 분당구 판교역로 166", "lat": 37.3952, "lon": 127.1114, "salary": "4,600만원"},
        {"job_id": "J12", "company_name": "파두 (FADU)", "company_type": "중견/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "C++ 펌웨어 SSD컨트롤러 반도체 RTL 회로설계", "address": "경기 성남시 분당구 판교로 242", "lat": 37.4018, "lon": 127.1052, "salary": "4,800만원"},
        {"job_id": "J13", "company_name": "SK케미칼", "company_type": "중견/대기업계열", "target_major": "화학공학과", "sig_category": "화학/소재", "required_skills": "고분자합성 공정제어 화학공학 유기화학 공정설계 품질관리", "address": "경기 성남시 분당구 판교로 310", "lat": 37.4045, "lon": 127.1070, "salary": "4,600만원"},
        {"job_id": "J14", "company_name": "마키나락스", "company_type": "강소기업", "target_major": "산업공학과", "sig_category": "AI/ML", "required_skills": "Python 공정최적화 이상탐지 산업공학 머신러닝 MLOps", "address": "경기 성남시 분당구 판교역로 230", "lat": 37.3998, "lon": 127.1100, "salary": "4,500만원"},
        {"job_id": "J15", "company_name": "한글과컴퓨터", "company_type": "중견/상장", "target_major": "디자인학과", "sig_category": "UI/UX 디자인", "required_skills": "UIUX 소프트웨어디자인 Figma 웹디자인 그래픽디자인", "address": "경기 성남시 분당구 대왕판교로 644", "lat": 37.3980, "lon": 127.1025, "salary": "4,000만원"},

        # --- [대전 유성/대덕연구개발특구 - 바이오, 화학, 로봇] ---
        {"job_id": "J16", "company_name": "알테오젠 (AlteoGen)", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "단백질공학 유전자재조합 PCR 바이오의약품 HPLC 분석화학", "address": "대전 유성구 문지로 281-25", "lat": 36.3980, "lon": 127.3995, "salary": "4,000만원"},
        {"job_id": "J17", "company_name": "한화솔루션 중앙연구소", "company_type": "중견/대기업계열", "target_major": "화학공학과", "sig_category": "화학/소재", "required_skills": "석유화학 고분자재료 촉매연구 화학공정 기기분석 소재개발", "address": "대전 유성구 가정로 136", "lat": 36.3780, "lon": 127.3680, "salary": "4,700만원"},
        {"job_id": "J18", "company_name": "바이오니아", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "분자진단 PCR 유전자시약 바이오의약품 유전자분석", "address": "대전 대덕구 문평서로 8-11", "lat": 36.4468, "lon": 127.4025, "salary": "3,800만원"},
        {"job_id": "J19", "company_name": "비전세미콘", "company_type": "강소기업", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "반도체제조장비 로봇제어 CAD 3D설계 PLC 자동화", "address": "대전 유성구 테크노2로 187", "lat": 36.4338, "lon": 127.3940, "salary": "3,800만원"},
        {"job_id": "J20", "company_name": "레고켐바이오", "company_type": "중견/상장", "target_major": "화학공학과", "sig_category": "바이오/제약", "required_skills": "의약화학 유기합성 약물전달 신약개발 분석화학 HPLC", "address": "대전 유성구 문지로 290", "lat": 36.3985, "lon": 127.4010, "salary": "4,200만원"},

        # --- [경기 수원/인천 송도 - 로봇, 바이오, 화학, 경영] ---
        {"job_id": "J21", "company_name": "두산로보틱스", "company_type": "중견/대기업계열", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "C++ ROS 로봇제어 역학설계 CAD 3D모데링 자동화", "address": "경기 수원시 영통구 삼성로 156", "lat": 37.2636, "lon": 127.0514, "salary": "4,500만원"},
        {"job_id": "J22", "company_name": "삼성바이오에피스", "company_type": "중견/대기업계열", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "바이오시밀러 세포주개발 배양공정 정제공정 HPLC", "address": "인천 연수구 송도바이오대로 107", "lat": 37.3820, "lon": 126.6680, "salary": "4,700만원"},
        {"job_id": "J23", "company_name": "경동나비엔", "company_type": "중견/상장", "target_major": "산업공학과", "sig_category": "품질/공정관리", "required_skills": "품질관리 공정개선 생산관리 6시그마 공정최적화 ISO", "address": "경기 평택시 서탄면 서탄로 205", "lat": 37.1180, "lon": 127.0510, "salary": "4,200만원"},
        {"job_id": "J24", "company_name": "동아에스티", "company_type": "중견/상장", "target_major": "화학공학과", "sig_category": "바이오/제약", "required_skills": "제약공정 의약품제조 제형연구 품질보증 GMP 분석화학", "address": "인천 연수구 지식기반로 45", "lat": 37.3710, "lon": 126.6520, "salary": "4,200만원"},
        {"job_id": "J25", "company_name": "고영테크놀러지", "company_type": "중견/상장", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "3D검사장비 C++ 로봇제어 비전검사 정밀구동 CAD", "address": "경기 용인시 수지구 신수로 767", "lat": 37.3275, "lon": 127.1022, "salary": "4,400만원"}
    ])

if "students_db" not in st.session_state:
    st.session_state.students_db = pd.DataFrame([
        {"student_id": "S01", "name": "김철수", "major": "컴퓨터공학과", "sig_interests": "AI/ML, 데이터 엔지니어링", "skills": "Python PyTorch 딥러닝 ComputerVision OpenCV 데이터분석", "email": "chulsoo.kim@univ.ac.kr"},
        {"student_id": "S02", "name": "이영희", "major": "경영학과", "sig_interests": "마케팅/기획", "skills": "퍼포먼스마케팅 데이터분석 SQL 서비스기획 마케팅전략 GA4", "email": "yh.lee@univ.ac.kr"},
        {"student_id": "S03", "name": "박민수", "major": "산업공학과", "sig_interests": "데이터 엔지니어링, 품질/공정관리", "skills": "SQL Python 데이터기반기획 프로세스최적화 6시그마 품질관리", "email": "ms.park@univ.ac.kr"},
        {"student_id": "S04", "name": "정수진", "major": "디자인학과", "sig_interests": "UI/UX 디자인", "skills": "Figma UIUX디자인 프로토타이핑 유저리서치 서비스디자인 웹디자인", "email": "sj.jung@univ.ac.kr"},
        {"student_id": "S05", "name": "최현우", "major": "화학공학과", "sig_interests": "화학/소재, 바이오/제약", "skills": "고분자합성 화학공정 유기화학 HPLC 기기분석 품질보증", "email": "hw.choi@univ.ac.kr"}
    ])

# 지오코더 (주소 -> 위경도 변환)
@st.cache_data
def geocode_address(address_str):
    try:
        geolocator = Nominatim(user_agent="sigs_job_platform_v3")
        location = geolocator.geocode(address_str)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    return None, None

# 3. 메인 타이틀
st.title("🗺️ SGIS 공간위치 기반 대학생 ↔ 강소기업 정밀 매칭 플랫폼")
st.caption("인문/사회, 공학, 디자인, 바이오 등 다양한 전공 기반의 알짜 강소·중견기업 공고와 내 위치 중심 매칭을 지원합니다.")

# 사이드바 모드 전환
mode = st.sidebar.radio(
    "📌 서비스 모드 선택",
    ["🎓 대학생 (내 근처 맞춤 기업 지도 탐색)", "🏢 기업 담당자 (공고 등록 & 인재 검색)"]
)

st.sidebar.markdown("---")

# ==========================================
# 모드 1: 대학생 모드 (다양한 전공 선택 및 공간 매칭)
# ==========================================
if mode == "🎓 대학생 (내 근처 맞춤 기업 지도 탐색)":
    st.header("🎓 SGIS 위치 기반 내 근처 기업 탐색 및 직무 매칭")
    
    st.sidebar.subheader("👤 대학생 프로필 & 직접 위치 입력")
    student_name = st.sidebar.text_input("이름", value="김철수")
    
    # 8개 전공 라인업 확충
    student_major = st.sidebar.selectbox(
        "전공 선택", 
        ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과", "산업공학과", "화학공학과", "디자인학과"]
    )
    
    input_address = st.sidebar.text_input(
        "📍 내 현재 주소/거주지 직접 입력",
        value="서울시 강남구 역삼동",
        help="예시: 서울시 강남구, 경기 판교역, 대전 유성구, 인천 송도 등"
    )
    
    user_lat, user_lon = geocode_address(input_address)
    
    if user_lat is None or user_lon is None:
        st.sidebar.warning("⚠️ 주소를 찾을 수 없어 기본 위치(서울 강남)로 설정됩니다.")
        user_lat, user_lon = 37.4979, 127.0276
    else:
        st.sidebar.success(f"📍 위치 인식 완료: ({round(user_lat, 4)}, {round(user_lon, 4)})")

    max_distance_km = st.sidebar.slider("📏 통근 가능 최대 반경 범위 (km)", min_value=1, max_value=50, value=20)
    
    # 다양해진 직무 카테고리
    student_interests = st.sidebar.multiselect(
        "SIGS 관심 분야 선택",
        ["AI/ML", "데이터 엔지니어링", "임베디드", "바이오/제약", "클라우드", "로봇/제어", "마케팅/기획", "UI/UX 디자인", "화학/소재", "품질/공정관리"],
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
        st.subheader(f"📍 '{input_address}' 주변 기업 지도 ({len(filtered_jobs)}개 검색됨)")
        
        m = folium.Map(location=[user_lat, user_lon], zoom_start=11)
        
        # 내 위치 마커
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

        # 반경 내 기업 파란색 마커 핀
        for _, row in filtered_jobs.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"<b>{row['company_name']}</b><br>선호 전공: {row['target_major']}<br>분야: {row['sig_category']}<br>거리: {row['distance_km']}km",
                tooltip=f"{row['company_name']} ({row['distance_km']}km)",
                icon=folium.Icon(color="blue", icon="building", prefix="fa")
            ).add_to(m)

        st_folium(m, width="100%", height=480)

    with col2:
        st.subheader(f"🎯 반경 {max_distance_km}km 내 추천 기업 리스트")
        
        if filtered_jobs.empty:
            st.warning(f"선택한 위치에서 반경 {max_distance_km}km 내에 등록된 기업이 없습니다. 반경 범위를 넓혀보세요.")
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
                    "선호 전공": row['target_major'],
                    "거리(km)": f"{row['distance_km']} km",
                    "SIG 분야": row['sig_category'],
                    "요구 스킬": row['required_skills'],
                    "연봉": row['salary']
                })

            res_df = pd.DataFrame(results).sort_values(by="매칭 점수", ascending=False)
            
            top_company = res_df.iloc[0]
            st.success(f"🔥 **{student_major}** 맞춤 추천 1순위: **{top_company['기업명']}** (거리: {top_company['거리(km)']}, 매칭률: **{top_company['매칭 점수']}점**)")
            
            st.dataframe(
                res_df,
                column_config={"매칭 점수": st.column_config.NumberColumn(format="%d점")},
                use_container_width=True,
                height=380,
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
                target_major = st.selectbox("우선 선호 전공", ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과", "산업공학과", "화학공학과", "디자인학과"])
                address = st.text_input("기업 도로명 주소", value="서울시 마포구 상암동")
            with col2:
                sig_category = st.selectbox("SIGS 직무 분야", ["AI/ML", "데이터 엔지니어링", "임베디드", "바이오/제약", "클라우드", "로봇/제어", "마케팅/기획", "UI/UX 디자인", "화학/소재", "품질/공정관리"])
                required_skills = st.text_input("필요 역량/스킬 키워드 (공백 구분)", value="Python C++ 딥러닝")
                salary = st.text_input("연봉 조건", value="4,000만원")
            
            submit_btn = st.form_submit_button("📢 SGIS 지도 등록 공고 제출")
            
            if submit_btn:
                if not company_name or not required_skills:
                    st.error("기업명과 필요 역량 키워드는 필수 사항입니다.")
                else:
                    lat, lon = geocode_address(address)
                    if lat is None or lon is None:
                        lat, lon = 37.5665, 126.9780
                    
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
                    st.success(f"✅ [{company_name}] 공고가 지도 좌표 변환되어 정상 등록되었습니다!")

    with tab2:
        st.subheader("🎯 기업 맞춤형 등록 인재 조회")
        st.dataframe(st.session_state.students_db, use_container_width=True, hide_index=True)
