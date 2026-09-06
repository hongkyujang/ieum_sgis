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

# 2. 세션 상태(Session State) 데이터베이스 초기화 (지역별 10개 이상 풍부한 DB 구축)
if "jobs_db" not in st.session_state:
    st.session_state.jobs_db = pd.DataFrame([
        # --- [권역 1: 서울 강남/성수/마포 권역 (10개)] ---
        {"job_id": "J01", "company_name": "루닛 (Lunit)", "company_type": "강소/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python PyTorch 딥러닝 의료영상분석 ComputerVision OpenCV", "address": "서울 강남구 테헤란로 211", "lat": 37.5032, "lon": 127.0416, "salary": "4,500만원"},
        {"job_id": "J02", "company_name": "센드버드 (Sendbird)", "company_type": "유니콘/강소", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Python Go AWS Docker Kubernetes API 백엔드", "address": "서울 강남구 테헤란로 142", "lat": 37.5002, "lon": 127.0365, "salary": "4,800만원"},
        {"job_id": "J03", "company_name": "원티드랩 (Wanted Lab)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python SQL 데이터엔지니어링 Spark ETL BigQuery", "address": "서울 송파구 올림픽로 300", "lat": 37.5137, "lon": 127.1042, "salary": "4,200만원"},
        {"job_id": "J04", "company_name": "몰로코 (Moloco)", "company_type": "유니콘/강소", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python C++ 머신러닝 데이터분석 빅데이터 타겟팅", "address": "서울 강남구 테헤란로 521", "lat": 37.5088, "lon": 127.0608, "salary": "5,200만원"},
        {"job_id": "J05", "company_name": "뤼이드 (Riiid)", "company_type": "강소기업", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python PyTorch NLP 자연어처리 딥러닝 알고리즘", "address": "서울 강남구 영동대로 517", "lat": 37.5126, "lon": 127.0588, "salary": "4,400만원"},
        {"job_id": "J06", "company_name": "쏘카 (SOCAR)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python SQL 데이터분석 머신러닝 모빌리티 최적화", "address": "서울 성동구 아차산로 6", "lat": 37.5467, "lon": 127.0435, "salary": "4,300만원"},
        {"job_id": "J07", "company_name": "직방 (Zigbang)", "company_type": "강소기업", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Python Node.js AWS MSA Docker 백엔드 웹개발", "address": "서울 서초구 서초대로 398", "lat": 37.4975, "lon": 127.0253, "salary": "4,100만원"},
        {"job_id": "J08", "company_name": "버킷플레이스 (오늘의집)", "company_type": "유니콘/강소", "target_major": "컴퓨터공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python SQL 데이터엔지니어링 ETL Airflow Spark", "address": "서울 서초구 서초대로 301", "lat": 37.4988, "lon": 127.0160, "salary": "4,600만원"},
        {"job_id": "J09", "company_name": "당근 (Daangn)", "company_type": "유니콘/강소", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python PyTorch 추천시스템 데이터분석 머신러닝", "address": "서울 서초구 강남대로 373", "lat": 37.4939, "lon": 127.0290, "salary": "4,700만원"},
        {"job_id": "J10", "company_name": "야놀자 (Yanolja)", "company_type": "중견/유니콘", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Java Spring Python AWS Kubernetes Docker MSA", "address": "서울 강남구 테헤란로 108", "lat": 37.4988, "lon": 127.0289, "salary": "4,500만원"},

        # --- [권역 2: 경기 판교/분당 IT 밸리 (10개)] ---
        {"job_id": "J11", "company_name": "리벨리온 (Rebellions)", "company_type": "강소/스타트업", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "C++ C 임베디드 FPGA SoC 회로설계 반도체설계 RTL", "address": "경기 성남시 분당구 판교역로 166", "lat": 37.3952, "lon": 127.1114, "salary": "4,600만원"},
        {"job_id": "J12", "company_name": "파두 (FADU)", "company_type": "중견/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "C++ 펌웨어 SSD컨트롤러 반도체 RTL 회로설계", "address": "경기 성남시 분당구 판교로 242", "lat": 37.4018, "lon": 127.1052, "salary": "4,800만원"},
        {"job_id": "J13", "company_name": "안랩 (AhnLab)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "C C++ Python 보안 클라우드보안 네트워크 악성코드분석", "address": "경기 성남시 분당구 판교역로 220", "lat": 37.4001, "lon": 127.1105, "salary": "4,200만원"},
        {"job_id": "J14", "company_name": "웹젠 (WEBZEN)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "C++ Python 게임서버 딥러닝 그래픽스 OpenCV", "address": "경기 성남시 분당구 판교로 242", "lat": 37.4015, "lon": 127.1060, "salary": "4,100만원"},
        {"job_id": "J15", "company_name": "가비아 (Gabia)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Linux Python OpenStack AWS Docker 네트워크 시스템운영", "address": "경기 성남시 분당구 대왕판교로 660", "lat": 37.4008, "lon": 127.1040, "salary": "3,900만원"},
        {"job_id": "J16", "company_name": "원스토어 (ONE store)", "company_type": "중견/강소", "target_major": "컴퓨터공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python SQL BigQuery Spark ETL 데이터엔지니어링", "address": "경기 성남시 분당구 판교역로 235", "lat": 37.4005, "lon": 127.1118, "salary": "4,300만원"},
        {"job_id": "J17", "company_name": "마키나락스 (MakinaRocks)", "company_type": "강소기업", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python PyTorch 산업용AI 이상탐지 머신러닝 MLOps", "address": "경기 성남시 분당구 판교역로 230", "lat": 37.3998, "lon": 127.1100, "salary": "4,500만원"},
        {"job_id": "J18", "company_name": "솔트룩스 (Saltlux)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python NLP 대형언어모델 LLM 자연어처리 PyTorch", "address": "경기 성남시 분당구 판교로 255", "lat": 37.4030, "lon": 127.1080, "salary": "4,000만원"},
        {"job_id": "J19", "company_name": "한콤 (Hancom)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "C++ Python AI문서분석 OCR 딥러닝 소프트웨어개발", "address": "경기 성남시 분당구 대왕판교로 644", "lat": 37.3980, "lon": 127.1025, "salary": "4,100만원"},
        {"job_id": "J20", "company_name": "윈스 (WINS)", "company_type": "중견/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "C C++ 네트워크 임베디드 침입탐지 방화벽 펌웨어", "address": "경기 성남시 분당구 판교로 228", "lat": 37.4022, "lon": 127.1072, "salary": "4,000만원"},

        # --- [권역 3: 대전 유성/대덕연구개발특구 (10개)] ---
        {"job_id": "J21", "company_name": "알테오젠 (AlteoGen)", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "단백질공학 유전자재조합 PCR 바이오의약품 HPLC 분석화학", "address": "대전 유성구 문지로 281-25", "lat": 36.3980, "lon": 127.3995, "salary": "4,000만원"},
        {"job_id": "J22", "company_name": "바이오니아 (Bioneer)", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "분자진단 PCR 유전자시약 바이오의약품 유전자분석", "address": "대전 대덕구 문평서로 8-11", "lat": 36.4468, "lon": 127.4025, "salary": "3,800만원"},
        {"job_id": "J23", "company_name": "레고켐바이오 (LegoChem)", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "ADC 항체유기합성 의약화학 바이오의약품 HPLC", "address": "대전 유성구 문지로 290", "lat": 36.3985, "lon": 127.4010, "salary": "4,200만원"},
        {"job_id": "J24", "company_name": "팹트론 (Peptron)", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "펩타이드 약물전달시스템 스마트포인트 단백질분석", "address": "대전 유성구 유성대로 1628", "lat": 36.4215, "lon": 127.3820, "salary": "3,900만원"},
        {"job_id": "J25", "company_name": "지노믹트리 (Genomictree)", "company_type": "강소/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "암분자진단 메틸화DNA NGS 유전자분석 PCR", "address": "대전 유성구 유성대로 1646", "lat": 36.4230, "lon": 127.3828, "salary": "3,950만원"},
        {"job_id": "J26", "company_name": "아이엔테라퓨틱스", "company_type": "강소기업", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "신약개발 이온채널 약리학 전임상 약효평가", "address": "대전 유성구 대덕대로 593", "lat": 36.3810, "lon": 127.3840, "salary": "4,100만원"},
        {"job_id": "J27", "company_name": "인비즈넷 (Inbiznet)", "company_type": "강소기업", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python C++ 영상처리 스마트시티 객체인지 OpenCV", "address": "대전 유성구 테크노1로 75", "lat": 36.4280, "lon": 127.3920, "salary": "3,700만원"},
        {"job_id": "J28", "company_name": "트루윈 (Truwin)", "company_type": "중견/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "IR센서 열화상 센서응용회로 임베디드 C++ 회로설계", "address": "대전 유성구 테크노2로 210", "lat": 36.4350, "lon": 127.3955, "salary": "3,850만원"},
        {"job_id": "J29", "company_name": "비전세미콘 (Vision Semiconductor)", "company_type": "강소기업", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "반도체제조장비 로봇제어 CAD 3D설계 PLC 자동화", "address": "대전 유성구 테크노2로 187", "lat": 36.4338, "lon": 127.3940, "salary": "3,800만원"},
        {"job_id": "J30", "company_name": "세트렉아이 (Satrec Initiative)", "company_type": "중견/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "위성시스템 인공위성 임베디드 C++ FPGA 신호처리", "address": "대전 유성구 유성대로 1559", "lat": 36.4150, "lon": 127.3780, "salary": "4,300만원"},

        # --- [권역 4: 경기 수원/인천/송도 권역 (10개)] ---
        {"job_id": "J31", "company_name": "두산로보틱스 (Doosan Robotics)", "company_type": "중견/대기업계열", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "C++ ROS 로봇제어 역학설계 CAD 3D모데링 자동화", "address": "경기 수원시 영통구 삼성로 156", "lat": 37.2636, "lon": 127.0514, "salary": "4,500만원"},
        {"job_id": "J32", "company_name": "고영테크놀러지 (Koh Young)", "company_type": "중견/상장", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "3D검사장비 C++ 로봇제어 비전검사 정밀구동 CAD", "address": "경기 용인시 수지구 신수로 767", "lat": 37.3275, "lon": 127.1022, "salary": "4,400만원"},
        {"job_id": "J33", "company_name": "에스에프에이 (SFA)", "company_type": "중견/상장", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "물류자동화 스마트팩토리 PLC 로봇제어 CAD 3D설계", "address": "경기 화성시 동탄기흥로 580", "lat": 37.2150, "lon": 127.0980, "salary": "4,300만원"},
        {"job_id": "J34", "company_name": "에스엠코어 (SMCore)", "company_type": "중견/상장", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "자동창고 물류로봇 제어프로그래밍 C++ PLC CAD", "address": "경기 안성시 공도읍 기업단지로 62", "lat": 37.0010, "lon": 127.1420, "salary": "4,100만원"},
        {"job_id": "J35", "company_name": "삼성바이오에피스", "company_type": "중견/대기업계열", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "바이오시밀러 세포주개발 배양공정 정제공정 HPLC", "address": "인천 연수구 송도바이오대로 107", "lat": 37.3820, "lon": 126.6680, "salary": "4,700만원"},
        {"job_id": "J36", "company_name": "동아에스티 (Dong-A ST)", "company_type": "중견/상장", "target_major": "생명공학과", "sig_category": "바이오/제약", "required_skills": "신약연구 약리독성 약동학 유전자재조합 PCR 분석화학", "address": "인천 연수구 지식기반로 45", "lat": 37.3710, "lon": 126.6520, "salary": "4,200만원"},
        {"job_id": "J37", "company_name": "엠씨넥스 (MCNEX)", "company_type": "중견/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "카메라모듈 차량용카메라 자율주행센서 임베디드 C++", "address": "인천 연수구 송도과학로 16번길 13-25", "lat": 37.3812, "lon": 126.6590, "salary": "4,000만원"},
        {"job_id": "J38", "company_name": "캠시스 (Camsys)", "company_type": "중견/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "카메라모듈 초소형전기차 BMS 임베디드 회로설계 C++", "address": "인천 연수구 벤처로 108", "lat": 37.3880, "lon": 126.6410, "salary": "3,900만원"},
        {"job_id": "J39", "company_name": "유진로봇 (Yujin Robot)", "company_type": "중견/상장", "target_major": "기계공학과", "sig_category": "로봇/제어", "required_skills": "자율주행로봇 ROS SLAM C++ LiDAR 로봇제어", "address": "인천 연수구 하모니로 187번길 33", "lat": 37.3850, "lon": 126.6380, "salary": "4,150만원"},
        {"job_id": "J40", "company_name": "아진엑스텍 (AJINEXTEK)", "company_type": "강소/상장", "target_major": "전자공학과", "sig_category": "임베디드", "required_skills": "모션제어칩 임베디드모션 C++ FPGA 로봇제어기", "address": "경기 부천시 석천로 397", "lat": 37.5180, "lon": 126.7720, "salary": "3,850만원"},

        # --- [권역 5: 서울 구로/금천 IT 디지털단지 (10개)] ---
        {"job_id": "J41", "company_name": "(주)감마데이터", "company_type": "중소/강소", "target_major": "컴퓨터공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python SQL 데이터엔지니어링 ETL Spark BigQuery", "address": "서울 금천구 가산디지털1로 168", "lat": 37.4800, "lon": 126.8820, "salary": "3,800만원"},
        {"job_id": "J42", "company_name": "엠로 (emro)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "Python Java AI구매망분석 머신러닝 WebApp", "address": "서울 영등포구 영등포로 150", "lat": 37.5210, "lon": 126.8910, "salary": "4,100만원"},
        {"job_id": "J43", "company_name": "웹케시 (Webcash)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Java Spring 핀테크 API 백엔드 AWS SQL", "address": "서울 영등포구 양평로 22길 21", "lat": 37.5370, "lon": 126.8960, "salary": "4,000만원"},
        {"job_id": "J44", "company_name": "쿠콘 (Coocon)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python Java 마이데이터 API 데이터수집 ETL", "address": "서울 영등포구 양평로 22길 21", "lat": 37.5372, "lon": 126.8962, "salary": "4,150만원"},
        {"job_id": "J45", "company_name": "모바일리더 (MobileLeader)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "C++ Python OCR 문서인식 딥러닝 영상처리", "address": "서울 서초구 반포대로 22", "lat": 37.4820, "lon": 127.0090, "salary": "3,900만원"},
        {"job_id": "J46", "company_name": "포비즈코리아", "company_type": "강소기업", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "PHP Python AWS Docker 이커머스 솔루션개발", "address": "서울 구로구 디지털로 26길 123", "lat": 37.4850, "lon": 126.8910, "salary": "3,750만원"},
        {"job_id": "J47", "company_name": "데이타솔루션", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "데이터 엔지니어링", "required_skills": "Python R SPSS 데이터분석 빅데이터 Spark", "address": "서울 강남구 언주로 620", "lat": 37.5120, "lon": 127.0380, "salary": "3,950만원"},
        {"job_id": "J48", "company_name": "지란지교시큐리티", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "C++ Python 문서보안 메일보안 클라우드보안 Linux", "address": "서울 송파구 정의로 8길 9", "lat": 37.4860, "lon": 127.1220, "salary": "4,000만원"},
        {"job_id": "J49", "company_name": "시큐센 (Secusen)", "company_type": "강소/상장", "target_major": "컴퓨터공학과", "sig_category": "AI/ML", "required_skills": "생체인증 AI전자서명 Python C++ 보안알고리즘", "address": "서울 금천구 가산디지털1로 145", "lat": 37.4780, "lon": 126.8810, "salary": "3,800만원"},
        {"job_id": "J50", "company_name": "핑거 (Finger)", "company_type": "중견/상장", "target_major": "컴퓨터공학과", "sig_category": "클라우드", "required_skills": "Java Spring 스마트뱅킹 모바일블록체인 AWS API", "address": "서울 영등포구 의사당대로 143", "lat": 37.5200, "lon": 126.9270, "salary": "4,050만원"}
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

# 지오코더(주소 -> 위경도 변환)
@st.cache_data
def geocode_address(address_str):
    try:
        geolocator = Nominatim(user_agent="sigs_job_platform_v2")
        location = geolocator.geocode(address_str)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    return None, None

# 3. 메인 타이틀
st.title("🗺️ SGIS기반 대학생 ↔ 기업 정밀 매칭 플랫폼")
st.caption("전국 권역별 50개 이상의 강소·중견기업 공고를 바탕으로 내 주소 기반 근거리 알짜 기업을 시각화 및 직무 매칭합니다.")

# 사이드바 모드 전환
mode = st.sidebar.radio(
    "📌 서비스 모드 선택",
    ["🎓 대학생 (내 근처 맞춤 기업 지도 탐색)", "🏢 기업 담당자 (공고 등록 & 인재 검색)"]
)

st.sidebar.markdown("---")

# ==========================================
# 모드 1: 대학생 모드 (지역별 10개 이상 기업 데이터 기반 연산)
# ==========================================
if mode == "🎓 대학생 (내 근처 맞춤 기업 지도 탐색)":
    st.header("🎓 SGIS 위치 기반 내 근처 기업 탐색 및 직무 매칭")
    
    st.sidebar.subheader("👤 대학생 프로필 & 직접 위치 입력")
    student_name = st.sidebar.text_input("이름", value="김철수")
    student_major = st.sidebar.selectbox(
        "전공 선택", 
        ["컴퓨터공학과", "전자공학과", "생명공학과", "기계공학과", "경영학과"]
    )
    
    # 주소 입력 예시 가이드 제공
    input_address = st.sidebar.text_input(
        "📍 내 현재 주소/거주지 직접 입력",
        value="서울시 강남구 역삼동",
        help="예시: 서울시 강남구, 경기 판교역, 대전 유성구, 인천 송도, 서울 가산동 등"
    )
    
    user_lat, user_lon = geocode_address(input_address)
    
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

    # 거리 계산 및 공간 필터링
    jobs_df = st.session_state.jobs_db.copy()
    
    def calc_dist(row):
        return round(geodesic((user_lat, user_lon), (row['lat'], row['lon'])).km, 2)

    jobs_df['distance_km'] = jobs_df.apply(calc_dist, axis=1)
    filtered_jobs = jobs_df[jobs_df['distance_km'] <= max_distance_km].copy()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader(f"📍 '{input_address}' 주변 기업 지도 ({len(filtered_jobs)}개 검색됨)")
        
        m = folium.Map(location=[user_lat, user_lon], zoom_start=11)
        
        # 내 위치 빨간색 마커
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
                popup=f"<b>{row['company_name']}</b><br>{row['sig_category']}<br>주소: {row['address']}<br>거리: {row['distance_km']}km",
                tooltip=f"{row['company_name']} ({row['distance_km']}km)",
                icon=folium.Icon(color="blue", icon="building", prefix="fa")
            ).add_to(m)

        st_folium(m, width="100%", height=480)

    with col2:
        st.subheader(f"🎯 반경 {max_distance_km}km 내 매칭 순위")
        
        if filtered_jobs.empty:
            st.warning(f"입력하신 위치에서 반경 {max_distance_km}km 내에 등록된 기업이 없습니다. 반경 범위를 넓혀보세요.")
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
            st.success(f"🔥 최우선 추천: **{top_company['기업명']}** (거리: {top_company['거리(km)']}, 매칭률: **{top_company['매칭 점수']}점**)")
            
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
