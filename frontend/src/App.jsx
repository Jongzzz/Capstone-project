import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ReferenceLine, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';

function App() {
  // ... (기존 인증 관련 상태는 동일하므로 생략)
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [currentUser, setCurrentUser] = useState(null);
  const [authData, setAuthData] = useState({ role: '의사', licenseNumber: '', password: '', confirmPassword: '', name: '', accessCode: '' });
  const [usersDB, setUsersDB] = useState(() => { const savedUsers = localStorage.getItem('coreTriageUsersDB'); return savedUsers ? JSON.parse(savedUsers) : []; });

  // [기존 핸들러 생략]
  const handleAuthChange = (e) => { setAuthData({ ...authData, [e.target.name]: e.target.value }); };
  const handleSignup = (e) => { e.preventDefault(); const MASTER_CODE = "CORE-2026"; if (authData.accessCode !== MASTER_CODE) return alert("원내 인가 코드가 일치하지 않습니다."); if (authData.password !== authData.confirmPassword) return alert("비밀번호가 일치하지 않습니다."); const newUser = { role: authData.role, name: authData.name, licenseNumber: authData.licenseNumber, password: authData.password }; const updatedDB = [...usersDB, newUser]; setUsersDB(updatedDB); localStorage.setItem('coreTriageUsersDB', JSON.stringify(updatedDB)); alert("등록 완료!"); setAuthMode('login'); };
  const handleLogin = (e) => { e.preventDefault(); const user = usersDB.find(u => u.licenseNumber === authData.licenseNumber && u.password === authData.password); if (user) { setCurrentUser(user); setIsLoggedIn(true); localStorage.setItem('coreTriageAuthToken', JSON.stringify(user)); } else { alert("로그인 실패"); } };
  const handleLogout = () => { setIsLoggedIn(false); setCurrentUser(null); localStorage.removeItem('coreTriageAuthToken'); };
  useEffect(() => { const token = localStorage.getItem('coreTriageAuthToken'); if (token) { setCurrentUser(JSON.parse(token)); setIsLoggedIn(true); } }, []);

  // ==========================================
  // 🚑 대시보드(CDSS) 핵심 로직 (수정됨)
  // ==========================================
  const [activeTab, setActiveTab] = useState('triage');
  const [patientHistory, setPatientHistory] = useState(() => {
    const saved = localStorage.getItem('coreTriageHistory');
    return saved ? JSON.parse(saved) : [];
  });

  const [formData, setFormData] = useState({
    patient_name: '', chief_complaint: '흉통/심장질환', age: 70,
    temperature: 36.5, heart_rate: 80, resp_rate: 20,
    o2sat: 98, sbp: 120, dbp: 80, pain_score: 0
  });

  const [selectedPatient, setSelectedPatient] = useState(null);
  const [loading, setLoading] = useState(false);

  const complaintOptions = ['흉통/심장질환', '호흡곤란', '복통', '두통/뇌졸중', '외상/출혈', '발열', '기타'];

  const maskName = (name) => { if (!name) return ""; return name.charAt(0) + '*'.repeat(name.length - 1); };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: (name === 'patient_name' || name === 'chief_complaint') ? value : (value === '' ? '' : Number(value)) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      // ✅ 1. Render 백엔드로 데이터 전송
      const response = await axios.post('https://c-o-r-e.onrender.com/api/triage/predict', formData);
      const newResult = response.data.data;

      // ✅ 2. 제미나이가 준 ai_briefing을 포함하여 데이터 저장
      const newPatientRecord = {
        id: newResult.patient_id, 
        name: newResult.patient_name, 
        cc: formData.chief_complaint,
        age: formData.age, 
        spo2: formData.o2sat, 
        sbp: formData.sbp,
        level: newResult.predicted_level, 
        score: newResult.risk_score, 
        time: newResult.timestamp,
        warnings: newResult.warnings, 
        xai_data: newResult.xai_data, 
        ai_briefing: newResult.ai_briefing, // 👈 렌더링을 위해 추가!
        isActive: true
      };
      
      const updatedHistory = [newPatientRecord, ...patientHistory];
      setPatientHistory(updatedHistory);
      localStorage.setItem('coreTriageHistory', JSON.stringify(updatedHistory));
      setSelectedPatient(newPatientRecord);
    } catch (error) { 
      alert("백엔드 연결 실패! Render 서버가 켜져 있는지(Live 상태인지) 확인해주세요."); 
      console.error(error); 
    }
    setLoading(false);
  };

  // [유틸리티 함수들]
  const handleDischarge = (id, e) => { e.stopPropagation(); const updatedHistory = patientHistory.map(p => p.id === id ? { ...p, isActive: false } : p ); setPatientHistory(updatedHistory); if (selectedPatient?.id === id) setSelectedPatient(null); };
  const getLevelStats = () => { /* 통계 로직 생략 */ return []; };
  const getLevelClass = (level) => `level-${level}`;
  const activeQueue = patientHistory.filter(p => p.isActive).sort((a, b) => b.score - a.score);

  // ==========================================
  // 3️⃣ 렌더링 화면 (중요 부분만 강조)
  // ==========================================
  if (!isLoggedIn) return (<div className="auth-wrapper"> {/* 로그인 화면 생략 */} </div>);

  return (
    <div className="app-layout">
      {/* 사이드바 생략 */}
      <div className="main-content">
        <div className="dashboard-header">
            <h1>CDSS Triage Engine</h1>
            <div>{currentUser.name} 님 접속 중</div>
        </div>
        
        {activeTab === 'triage' && (
          <div className="dashboard-grid-3">
            {/* [1컬럼] 입력 폼 */}
            <div className="dashboard-card">
              <div className="card-title">🩺 Patient Profile</div>
              <form onSubmit={handleSubmit} className="triage-form">
                <input type="text" name="patient_name" value={formData.patient_name} onChange={handleChange} className="form-input" placeholder="성함" required />
                {/* 주증상 선택기 생략 */}
                <button type="submit" disabled={loading} className="form-submit-button">
                  {loading ? '분석 중...' : '🚀 AI Triage 가동'}
                </button>
              </form>
            </div>

            {/* [2컬럼] 대기열 생략 (기존과 동일) */}

            {/* [3컬럼] AI 브리핑 출력 영역 (수정됨) */}
            <div className="analytics-column">
              <div className="dashboard-card xai-card">
                <div className="card-title">🦾 AI 임상 소견 (Gemini)</div>
                
                {selectedPatient ? (
                  <div className="report-content">
                    {/* 🤖 제미나이의 브리핑을 가장 잘 보이는 곳에 배치! */}
                    <div style={{ backgroundColor: '#F0F9FF', border: '1px solid #BAE6FD', padding: '15px', borderRadius: '10px', marginBottom: '15px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#0369A1', fontWeight: 800, marginBottom: '5px' }}>👨‍⚕️ AI 비서 브리핑:</div>
                        <div style={{ fontSize: '0.95rem', color: '#0C4A6E', lineHeight: '1.6', fontWeight: 600 }}>
                            {selectedPatient.ai_briefing || "브리핑을 생성하는 중입니다..."}
                        </div>
                    </div>

                    {/* XAI 차트 영역 */}
                    <div className="chart-container" style={{ height: '140px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={selectedPatient.xai_data} layout="vertical">
                                <XAxis type="number" hide />
                                <YAxis dataKey="name" type="category" width={40} style={{fontSize: '0.7rem'}} />
                                <Bar dataKey="value">
                                    {selectedPatient.xai_data?.map((entry, index) => <Cell key={index} fill={entry.value > 0 ? '#EF4444' : '#10B981'} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* 룰베이스 경고 영역 */}
                    <div style={{ marginTop: '10px' }}>
                        {selectedPatient.warnings?.map((warn, i) => (
                            <div key={i} style={{ fontSize: '0.8rem', color: '#991B1B', backgroundColor: '#FEF2F2', padding: '8px', borderRadius: '5px', marginBottom: '5px' }}>
                                🚨 {warn}
                            </div>
                        ))}
                    </div>
                  </div>
                ) : (
                  <div className="empty-state">환자를 클릭하면 AI 브리핑이 표시됩니다.</div>
                )}
              </div>
              {/* Fairness Audit 생략 */}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App;