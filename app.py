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
    page_title="SGIS 기반 대학생-기업 매칭 플랫폼", 
    page_icon="🗺️",
    layout="wide"
)

# 2. 세션 상태(Session State) 데이터베이스 초기화
if "jobs_db" not in st.session_state:
    st.session_state.jobs_db = pd.DataFrame([
        # --- [서울 권역] ---
        {"job_id": "J01", "company_name": "루닛 (Lunit)", "company_type": "강소/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python PyTorch 딥러닝 의료영상분석 ComputerVision OpenCV", "address": "서울 강남구 테헤란로 211", "lat": 37.5032, "lon": 127.0416, "salary": "4,500만원"},
        {"job_id": "J02", "company_name": "센드버드 (Sendbird)", "company_type": "유니콘/강소", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Python Go AWS Docker Kubernetes API 백엔드", "address": "서울 강남구 테헤란로 142", "lat": 37.5002, "lon": 127.0365, "salary": "4,800만원"},
        {"job_id": "J03", "company_name": "원티드랩 (Wanted Lab)", "company_type": "중견/상장", "target_major": "경영학과", "sig_category": "마케팅/기획", "required_skills": "퍼포먼스마케팅 데이터분석 SQL 서비스기획 마케팅전략 GA4", "address": "서울 송파구 올림픽로 300", "lat": 37.5137, "lon": 127.1042, "salary": "4,000만원"},
        {"job_id": "J04", "company_name": "몰로코 (Moloco)", "company_type": "유니콘/강소", "target_major": "산업공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python SQL 머신러닝 최적화 데이터분석 통계분석 빅데이터", "address": "서울 강남구 테헤란로 521", "lat": 37.5088, "lon": 127.0608, "salary": "5,000만원"},
        {"job_id": "J05", "company_name": "뤼이드 (Riiid)", "company_type": "강소기업", "target_major": "디자인학과", "sig_category": "UI/UX 디자인", "required_skills": "Figma UIUX디자인 프로토타이핑 유저리서치 서비스디자인", "address": "서울 강남구 영동대로 517", "lat": 37.5126, "lon": 127.0588, "salary": "4,200만원"},
        
        # --- [판교/대전/수원 권역] ---
        {"job_id": "J06", "company_name": "리벨리온 (Rebellions)", "company_type": "강소/스타트업", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "C++ C 임베디드 FPGA SoC 회로설계 반도체설계 RTL", "address": "경기 성남시 분당구 판교역로 166", "lat": 37.3952, "lon": 127.1114, "salary": "4,600만원"},
        {"job_id": "J07", "company_name": "알테오젠 (AlteoGen)", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "단백질공학 유전자재조합 PCR 바이오의약품 HPLC 분석화학", "address": "대전 유성구 문지로 281-25", "lat": 36.3980, "lon": 127.3995, "salary": "4,000만원"},
        {"job_id": "J08", "company_name": "두산로보틱스", "company_type": "중견/대기업계열", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "C++ ROS 로봇제어 역학설계 CAD 3D모데링 자동화", "address": "경기 수원시 영통구 삼성로 156", "lat": 37.2636, "lon": 127.0514, "salary": "4,500만원"},
        {"job_id": "J09", "company_name": "SK케미칼", "company_type": "중견/대기업계열", "target_major": "화학공학과", "sig_category": "화학/소재", "required_skills": "고분자합성 공정제어 화학공학 유기화학 공정설계 품질관리", "address": "경기 성남시 분당구 판교로 310", "lat": 37.4045, "lon": 127.1070, "salary": "4,600만원"}
    ])

# 등록 학생 DB 대폭 확충 (KeyError 방지를 위해 address 컬럼 보장)
if "students_db" not in st.session_state or "address" not in st.session_state.students_db.columns:
    st.session_state.students_db = pd.DataFrame([
        {"student_id": "S01", "name": "김철수", "major": "컴퓨터공학과", "sig_interests": "AI/ML, 데이터 엔지니어링", "skills": "Python PyTorch 딥러닝 ComputerVision OpenCV 데이터분석", "email": "chulsoo.kim@univ.ac.kr", "address": "서울시 강남구 역삼동"},
        {"student_id": "S02", "name": "이영희", "major": "경영학과", "sig_interests": "마케팅/기획", "skills": "퍼포먼스마케팅 데이터분석 SQL 서비스기획 마케팅전략 GA4", "email": "yh.lee@univ.ac.kr", "address": "서울시 송파구 잠실동"},
        {"student_id": "S03", "name": "박민수", "major": "산업공학과", "sig_interests": "데이터 엔지니어링, AI/ML", "skills": "SQL Python 데이터기반기획 프로세스최적화 6시그마 머신러닝 통계분석 빅데이터", "email": "ms.park@univ.ac.kr", "address": "경기 성남시 분당구"},
        {"student_id": "S04", "name": "정수진", "major": "디자인학과", "sig_interests": "UI/UX 디자인", "skills": "Figma UIUX디자인 프로토타이핑 유저리서치 서비스디자인 웹디자인", "email": "sj.jung@univ.ac.kr", "address": "서울시 마포구 상암동"},
        {"student_id": "S05", "name": "최현우", "major": "화학공학과", "sig_interests": "화학/소재, 바이오/제약", "skills": "고분자합성 화학공정 유기화학 HPLC 기기분석 품질보증 공정설계", "email": "hw.choi@univ.ac.kr", "address": "대전시 유성구 궁동"},
        {"student_id": "S06", "name": "강동원", "major": "전자공학과", "sig_interests": "임베디드", "skills": "C++ C 임베디드 FPGA SoC 회로설계 RTL 펌웨어", "email": "dw.kang@univ.ac.kr", "address": "경기 성남시 판교동"},
        {"student_id": "S07", "name": "한지민", "major": "생명공학과", "sig_interests": "바이오/제약", "skills": "단백질공학 유전자재조합 PCR 바이오의약품 HPLC 분석화학", "email": "jm.han@univ.ac.kr", "address": "대전시 유성구 봉명동"},
        {"student_id": "S08", "name": "윤서준", "major": "기계공학과", "sig_interests": "로봇/제어", "skills": "C++ ROS 로봇제어 역학설계 CAD 3D모데링 자동화 PLC", "email": "sj.yoon@univ.ac.kr", "address": "경기 수원시 영통구"}
    ])

# 스카우트 제안 내역 저장용 세션
if "scout_history" not in st.session_state:
    st.session_state.scout_history = []

# 학생 모드 검색 세션 상태 초기화
if "student_search_triggered" not in st.session_state:
    st.session_state.student_search_triggered = False

# 지오코더 (주소 -> 위경도 변환)
@st.cache_data
def geocode_address(address_str):
    try:
        geolocator = Nominatim(user_agent="sigs_job_platform_v5")
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
    ["🎓 대학생 (내 근처 맞춤 기업 지도 탐색)", "🏢 기업 담당자 (공고 등록 & 인재 발굴)"]
)

st.sidebar.markdown("---")

# ==========================================
# 모드 1: 대학생 모드
# ==========================================
if mode == "🎓 대학생 (내 근처 맞춤 기업 지도 탐색)":
    st.header("🎓 SGIS 위치 기반 내 근처 기업 탐색 및 직무 매칭")
    
    st.sidebar.subheader("👤 대학생 프로필 & 직접 위치 입력")
    student_name = st.sidebar.text_input("이름", value="김철수")
    student_major = st.sidebar.selectbox(
        "전공 선택", 
        ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과", "산업공학과", "화학공학과", "디자인학과"]
    )
    
    input_address = st.sidebar.text_input(
        "📍 내 현재 주소/거주지 직접 입력",
        value="서울시 강남구 역삼동"
    )
    
    max_distance_km = st.sidebar.slider("📏 통근 가능 최대 반경 범위 (km)", min_value=1, max_value=50, value=20)
    
    student_interests = st.sidebar.multiselect(
        "SIGS 관심 분야 선택",
        ["AI/ML", "데이터 엔지니어링", "임베디드", "바이오/제약", "클라우드", "로봇/제어", "마케팅/기획", "UI/UX 디자인", "화학/소재", "품질/공정관리"],
        default=["AI/ML", "데이터 엔지니어링"]
    )
    student_skills = st.sidebar.text_area(
        "보유 기술/역량 키워드 (공백 구분)",
        value="Python PyTorch 딥러닝 ComputerVision OpenCV 데이터분석"
    )

    # 🔍 검색 버튼 추가
    search_btn = st.sidebar.button("🔍 내 근처 맞춤 기업 검색", type="primary", use_container_width=True)

    if search_btn or st.session_state.student_search_triggered:
        st.session_state.student_search_triggered = True

        user_lat, user_lon = geocode_address(input_address)
        
        if user_lat is None or user_lon is None:
            st.sidebar.warning("⚠️ 주소를 찾을 수 없어 기본 위치(서울 강남)로 설정됩니다.")
            user_lat, user_lon = 37.4979, 127.0276
        else:
            st.sidebar.success(f"📍 위치 인식 완료: ({round(user_lat, 4)}, {round(user_lon, 4)})")

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
            
            folium.Marker(
                location=[user_lat, user_lon],
                popup=f"<b>[내 위치] {student_name}님</b><br>{input_address}",
                tooltip=f"내 위치 ({input_address})",
                icon=folium.Icon(color="red", icon="user", prefix="fa")
            ).add_to(m)

            folium.Circle(
                location=[user_lat, user_lon],
                radius=max_distance_km * 1000,
                color="#3186cc",
                fill=True,
                fill_opacity=0.1
            ).add_to(m)

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
                st.warning(f"선택한 위치에서 반경 {max_distance_km}km 내에 등록된 기업이 없습니다.")
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
                
                st.dataframe(res_df, column_config={"매칭 점수": st.column_config.NumberColumn(format="%d점")}, use_container_width=True, height=380, hide_index=True)
    else:
        st.info("👈 좌측 사이드바에서 프로필과 위치를 입력한 후 **'🔍 내 근처 맞춤 기업 검색'** 버튼을 눌러주세요.")

# ==========================================
# 모드 2: 기업 담당자 모드
# ==========================================
else:
    tab1, tab2, tab3 = st.tabs(["🔎 공고별 맞춤 인재 검색 & 스카우트", "➕ 신규 채용 공고 등록", "📫 입사 제안(Scout) 발송 내역"])
    
    # 탭 1: 고도화된 공고별 맞춤 인재 발굴
    with tab1:
        st.subheader("🎯 기업 공고 선택 기반 인재 역량 매칭 시스템")
        
        jobs_df = st.session_state.jobs_db
        if jobs_df.empty:
            st.info("등록된 기업 공고가 없습니다. 신규 공고를 먼저 등록해주세요.")
        else:
            company_list = jobs_df['company_name'].tolist()
            selected_company_name = st.selectbox("🏢 인재를 검색할 자사 채용 공고를 선택하세요", company_list)
            
            job_info = jobs_df[jobs_df['company_name'] == selected_company_name].iloc[0]
            
            st.info(f"📌 **[{job_info['company_name']}] 공고 조건** | 선호 전공: **{job_info['target_major']}** | 필수/우대 역량: `{job_info['required_skills']}` | 위치: {job_info['address']}")
            
            st.markdown("---")
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_major_only = st.checkbox("선호 전공자만 검색 (전공 필터링)", value=False)
            with col_f2:
                min_match_score = st.slider("최소 매칭 점수 기준 설정 (점)", 0, 100, 30)

            students_df = st.session_state.students_db.copy()
            
            # TF-IDF 기반 인재 역량 매칭 점수 계산
            all_texts = [job_info['required_skills']] + list(students_df['skills'])
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            job_vec = tfidf_matrix[0]
            student_vecs = tfidf_matrix[1:]
            skill_sims = cosine_similarity(job_vec, student_vecs).flatten()

            results = []
            for idx, (_, row) in enumerate(students_df.iterrows()):
                skill_score = skill_sims[idx] * 100
                major_score = 30 if row['major'] == job_info['target_major'] else 0
                sig_score = 20 if job_info['sig_category'] in row['sig_interests'] else 0
                
                final_score = (skill_score * 0.5) + major_score + sig_score
                
                # 안전한 address 컬럼 접근 (KeyError 방지)
                student_addr = row.get('address', None)
                if pd.notnull(student_addr) and student_addr != "":
                    st_lat, st_lon = geocode_address(student_addr)
                    if st_lat and st_lon:
                        dist_km = round(geodesic((job_info['lat'], job_info['lon']), (st_lat, st_lon)).km, 1)
                    else:
                        dist_km = "위치 미상"
                else:
                    dist_km = "위치 미상"

                results.append({
                    "student_id": row['student_id'],
                    "매칭 점수": round(final_score, 1),
                    "이름": row['name'],
                    "전공": row['major'],
                    "관심 분야": row['sig_interests'],
                    "보유 역량": row['skills'],
                    "거주지": student_addr if student_addr else "미입력",
                    "기업과의 거리": f"{dist_km} km" if isinstance(dist_km, float) else dist_km,
                    "이메일": row['email']
                })

            res_df = pd.DataFrame(results)
            
            if filter_major_only:
                res_df = res_df[res_df['전공'] == job_info['target_major']]
            
            res_df = res_df[res_df['매칭 점수'] >= min_match_score].sort_values(by="매칭 점수", ascending=False)

            st.markdown(f"### 🏆 [ {selected_company_name} ] 추천 인재 순위 ({len(res_df)}명 탐색됨)")
            
            if res_df.empty:
                st.warning("조건에 부합하는 등록 인재가 없습니다. 필터링 조건을 변경해보세요.")
            else:
                st.dataframe(
                    res_df[["매칭 점수", "이름", "전공", "관심 분야", "보유 역량", "기업과의 거리"]],
                    column_config={"매칭 점수": st.column_config.NumberColumn(format="%d점")},
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                st.markdown("### ✉️ 우수 인재 상세 보기 및 입사 제안(Scout) 발송")
                
                candidate_names = res_df['이름'].tolist()
                selected_candidate_name = st.selectbox("상세 프로필 확인 및 입사 제안할 인재를 선택하세요", candidate_names)
                
                candidate_info = res_df[res_df['이름'] == selected_candidate_name].iloc[0]
                
                with st.expander(f"👤 {candidate_info['이름']} 학생 상세 프로필 보기 (매칭 점수: {candidate_info['매칭 점수']}점)", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**전공:** {candidate_info['전공']}")
                        st.write(f"**거주지:** {candidate_info['거주지']} (직장과의 거리: {candidate_info['기업과의 거리']})")
                        st.write(f"**이메일:** {candidate_info['이메일']}")
                    with c2:
                        st.write(f"**관심 분야:** {candidate_info['관심 분야']}")
                        st.write(f"**보유 핵심 기술:** {candidate_info['보유 역량']}")
                    
                    st.markdown("#### 📩 영입 제안 메시지 작성")
                    default_msg = f"안녕하세요 {candidate_info['이름']}님, [{selected_company_name}] 채용 담당자입니다. 님께서 보유하신 {candidate_info['보유 역량']} 역량이 당사의 {job_info['sig_category']} 포지션에 매우 적합하다고 판단하여 면접을 제안합니다."
                    scout_msg = st.text_area("스카우트 제안 메시지 내용", value=default_msg, height=100)
                    
                    if st.button(f"🚀 {candidate_info['이름']} 학생에게 입사 제안(Scout) 메시지 전송"):
                        st.session_state.scout_history.append({
                            "company": selected_company_name,
                            "student_name": candidate_info['이름'],
                            "student_email": candidate_info['이메일'],
                            "match_score": candidate_info['매칭 점수'],
                            "message": scout_msg
                        })
                        st.success(f"✅ {candidate_info['이름']}님({candidate_info['이메일']})에게 입사 제안이 성공적으로 발송되었습니다!")

    # 탭 2: 신규 채용 공고 등록
    with tab2:
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

    # 탭 3: 입사 제안 발송 내역
    with tab3:
        st.subheader("📫 제안 발송 내역 및 상태 관리")
        if not st.session_state.scout_history:
            st.info("현재 발송된 입사 제안 내역이 없습니다.")
        else:
            scout_df = pd.DataFrame(st.session_state.scout_history)
            st.dataframe(scout_df, use_container_width=True, hide_index=True)
