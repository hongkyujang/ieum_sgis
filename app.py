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
            "
