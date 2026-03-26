import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ReferenceLine, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';

function App() {
  // ==========================================
  // 🔒 인증(Auth) 관련 상태 관리
  // ==========================================
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [currentUser, setCurrentUser] = useState(null);

  const [authData, setAuthData] = useState({
    role: '의사', licenseNumber: '', password: '', confirmPassword: '', name: '', accessCode: ''
  });

  const [usersDB, setUsersDB] = useState(() => {
    const savedUsers = localStorage.getItem('coreTriageUsersDB');
    return savedUsers ? JSON.parse(savedUsers) : [];
  });

  const handleAuthChange = (e) => {
    setAuthData({ ...authData, [e.target.name]: e.target.value });
  };

  const handleSignup = (e) => {
    e.preventDefault();
    const MASTER_CODE = "CORE-2026"; 
    if (authData.accessCode !== MASTER_CODE) return alert("원내 인가 코드가 일치하지 않습니다. 관리자에게 문의하세요.");
    if (authData.password !== authData.confirmPassword) return alert("비밀번호가 일치하지 않습니다.");
    
    if (usersDB.find(u => u.licenseNumber === authData.licenseNumber)) return alert("이미 등록된 면허/사번입니다.");

    const newUser = { role: authData.role, name: authData.name, licenseNumber: authData.licenseNumber, password: authData.password };
    const updatedDB = [...usersDB, newUser];
    
    setUsersDB(updatedDB);
    localStorage.setItem('coreTriageUsersDB', JSON.stringify(updatedDB));
    alert("의료진 등록이 완료되었습니다! 로그인해주세요.");
    setAuthMode('login');
    setAuthData({ ...authData, password: '', confirmPassword: '', accessCode: '' });
  };

  const handleLogin = (e) => {
    e.preventDefault();
    const user = usersDB.find(u => u.licenseNumber === authData.licenseNumber && u.password === authData.password);
    if (user) {
      setCurrentUser(user); setIsLoggedIn(true);
      localStorage.setItem('coreTriageAuthToken', JSON.stringify(user));
    } else { alert("면허번호 또는 비밀번호가 잘못되었습니다."); }
  };

  const handleLogout = () => {
    if(window.confirm("안전하게 로그아웃 하시겠습니까?")) {
      setIsLoggedIn(false); setCurrentUser(null);
      localStorage.removeItem('coreTriageAuthToken');
      setAuthData({ ...authData, password: '' });
    }
  };

  useEffect(() => {
    const token = localStorage.getItem('coreTriageAuthToken');
    if (token) { setCurrentUser(JSON.parse(token)); setIsLoggedIn(true); }
  }, []);

  // ==========================================
  // 🚑 대시보드(CDSS) 관련 상태 관리
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

  const maskName = (name) => {
    if (!name) return "";
    if (name.length <= 2) return name.charAt(0) + '*';
    return name.charAt(0) + '*'.repeat(name.length - 2) + name.charAt(name.length - 1);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === 'patient_name' || name === 'chief_complaint') {
      setFormData({ ...formData, [name]: value });
    } else {
      setFormData({ ...formData, [name]: value === '' ? '' : Number(value) });
    }
  };

  const handleComplaintSelect = (option) => { setFormData({ ...formData, chief_complaint: option }); };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/triage/predict', formData);
      const newResult = response.data.data;

      const newPatientRecord = {
        id: newResult.patient_id, name: newResult.patient_name, cc: formData.chief_complaint,
        age: formData.age, spo2: formData.o2sat, sbp: formData.sbp,
        level: newResult.predicted_level, score: newResult.risk_score, time: newResult.timestamp,
        warnings: newResult.warnings, xai_data: newResult.xai_data, isActive: true
      };
      
      const updatedHistory = [newPatientRecord, ...patientHistory];
      setPatientHistory(updatedHistory);
      localStorage.setItem('coreTriageHistory', JSON.stringify(updatedHistory));
      setSelectedPatient(newPatientRecord);
    } catch (error) { alert("백엔드 서버와 연결할 수 없습니다."); }
    setLoading(false);
  };

  const handleDischarge = (id, e) => {
    e.stopPropagation();
    const updatedHistory = patientHistory.map(p => p.id === id ? { ...p, isActive: false } : p );
    setPatientHistory(updatedHistory);
    localStorage.setItem('coreTriageHistory', JSON.stringify(updatedHistory));
    if (selectedPatient && selectedPatient.id === id) setSelectedPatient(null);
  };

  const handleExportCSV = () => {
    if (patientHistory.length === 0) return alert("추출할 데이터가 없습니다.");
    const BOM = '\uFEFF'; 
    const headers = ['환자번호', '성명', '주증상', '나이', 'SpO2(%)', '수축기혈압(mmHg)', '예측등급(Level)', '위험도점수', '분석일시', '상태'];
    const csvRows = [headers.join(',')];
    patientHistory.forEach(p => csvRows.push([p.id, p.name, p.cc, p.age, p.spo2, p.sbp, p.level, p.score, p.time, p.isActive ? '대기중' : '진료완료'].join(',')));
    const blob = new Blob([BOM + csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob);
    link.setAttribute("download", `CDSS_Triage_${new Date().getTime()}.csv`);
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
  };

  const getLevelStats = () => {
    const counts = { 1:0, 2:0, 3:0, 4:0, 5:0 };
    patientHistory.forEach(p => { if(counts[p.level] !== undefined) counts[p.level]++; });
    return [
      { name: 'Lv.1 소생', value: counts[1], fill: '#DC2626' }, { name: 'Lv.2 긴급', value: counts[2], fill: '#EA580C' },
      { name: 'Lv.3 응급', value: counts[3], fill: '#CA8A04' }, { name: 'Lv.4 준응급', value: counts[4], fill: '#16A34A' },
      { name: 'Lv.5 비응급', value: counts[5], fill: '#475569' },
    ].filter(stat => stat.value > 0);
  };

  const getLevelClass = (level) => {
    if (level === 1) return 'level-1'; if (level === 2) return 'level-2';
    if (level === 3) return 'level-3'; if (level === 4) return 'level-4'; return 'level-5';
  };

  const activeQueue = patientHistory.filter(p => p.isActive).sort((a, b) => b.score - a.score);

  // ==========================================
  // 1️⃣ 비로그인 시 렌더링 (Auth 화면)
  // ==========================================
  if (!isLoggedIn) {
    return (
      <div className="auth-wrapper">
        <div className="auth-card">
          <div className="auth-logo">🏥</div>
          <h1 className="auth-title">C.O.R.E CDSS</h1>
          <p className="auth-subtitle">의료진 전용 시스템입니다. 인증 후 접속해주세요.</p>

          {authMode === 'login' ? (
            <form className="auth-form" onSubmit={handleLogin}>
              <div className="form-group">
                <label className="form-label">면허 번호 / 사번</label>
                <input type="text" name="licenseNumber" value={authData.licenseNumber} onChange={handleAuthChange} className="auth-input" placeholder="의사/간호사 면허번호 입력" required />
              </div>
              <div className="form-group">
                <label className="form-label">비밀번호</label>
                <input type="password" name="password" value={authData.password} onChange={handleAuthChange} className="auth-input" placeholder="비밀번호 입력" required />
              </div>
              <button type="submit" className="auth-submit-btn">보안 접속 (Login)</button>
              <div className="auth-switch">의료진 등록이 안 되어 있으신가요? <span onClick={() => setAuthMode('signup')}>회원가입</span></div>
            </form>
          ) : (
            <form className="auth-form" onSubmit={handleSignup}>
              <div className="form-group" style={{ backgroundColor: '#FEF2F2', padding: '15px', borderRadius: '8px', border: '1px solid #FECACA', marginBottom: '10px' }}>
                <label className="form-label" style={{ color: '#DC2626' }}>🔒 원내 인가 코드 (보안)</label>
                <input type="password" name="accessCode" value={authData.accessCode} onChange={handleAuthChange} className="auth-input" placeholder="관리자에게 부여받은 코드 (CORE-2026)" style={{ borderColor: '#F87171' }} required />
              </div>
              <div className="form-group">
                <label className="form-label">직군 선택</label>
                <div className="auth-role-group">
                  <button type="button" className={`auth-role-btn ${authData.role === '의사' ? 'active' : ''}`} onClick={() => setAuthData({...authData, role: '의사'})}>👨‍⚕️ 의사 (MD)</button>
                  <button type="button" className={`auth-role-btn ${authData.role === '간호사' ? 'active' : ''}`} onClick={() => setAuthData({...authData, role: '간호사'})}>👩‍⚕️ 간호사 (RN)</button>
                </div>
              </div>
              <div className="form-group"><label className="form-label">성명</label><input type="text" name="name" value={authData.name} onChange={handleAuthChange} className="auth-input" placeholder="실명 입력" required /></div>
              <div className="form-group"><label className="form-label">면허 번호 (ID)</label><input type="text" name="licenseNumber" value={authData.licenseNumber} onChange={handleAuthChange} className="auth-input" placeholder="면허번호 입력" required /></div>
              <div className="form-group"><label className="form-label">비밀번호 설정</label><input type="password" name="password" value={authData.password} onChange={handleAuthChange} className="auth-input" minLength="6" required /></div>
              <div className="form-group"><label className="form-label">비밀번호 확인</label><input type="password" name="confirmPassword" value={authData.confirmPassword} onChange={handleAuthChange} className="auth-input" required /></div>
              <button type="submit" className="auth-submit-btn" style={{backgroundColor: '#10B981'}}>신규 의료진 등록</button>
              <div className="auth-switch">이미 계정이 있으신가요? <span onClick={() => setAuthMode('login')}>로그인하기</span></div>
            </form>
          )}
        </div>
      </div>
    );
  }

  // ==========================================
  // 2️⃣ 로그인 성공 시 렌더링 (메인 대시보드)
  // ==========================================
  return (
    <div className="app-layout">
      <div className="sidebar">
        <div className={`sidebar-icon ${activeTab === 'triage' ? 'active' : ''}`} onClick={() => setActiveTab('triage')} title="Triage 분석">🚑</div>
        <div className={`sidebar-icon ${activeTab === 'roster' ? 'active' : ''}`} onClick={() => setActiveTab('roster')} title="환자 목록">👥</div>
        <div className={`sidebar-icon ${activeTab === 'stats' ? 'active' : ''}`} onClick={() => setActiveTab('stats')} title="통계 대시보드">📊</div>
        
        <div className="user-profile">
          <div className="user-avatar" title={`${currentUser.role} ${currentUser.name}`}>{currentUser.name.charAt(0)}</div>
          <button onClick={handleLogout} className="logout-btn" title="안전하게 로그아웃">🔒</button>
        </div>
      </div>

      <div className="main-content">
        <div className="dashboard-header">
          <div className="header-title-group">
            <h1 style={{display:'inline-block'}}>CDSS Triage Engine</h1>
            <span className="header-subtitle">응급환자 실시간 중증도 분류 시스템</span>
          </div>
          <div style={{color: '#1E293B', fontWeight: 600, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px'}}>
             <span style={{backgroundColor: '#EFF6FF', color: '#2563EB', padding: '4px 8px', borderRadius: '6px', fontSize: '0.8rem'}}>{currentUser.role}</span>
             {currentUser.name} 님 접속 중
          </div>
        </div>
        
        {activeTab === 'triage' && (
          <div className="dashboard-grid-3">
            {/* [1컬럼] 입력 폼 */}
            <div className="dashboard-card">
              <div className="card-title-group"><div className="card-title">🩺 Patient Profile</div></div>
              <form onSubmit={handleSubmit} className="triage-form" style={{flex: 1, display: 'flex', flexDirection: 'column'}}>
                
                <div className="form-group">
                  <label className="form-label">환자 성명</label>
                  <input type="text" name="patient_name" value={formData.patient_name} onChange={handleChange} className="form-input" required />
                </div>
                
                <div className="form-group">
                  <label className="form-label">주증상 (C.C)</label>
                  <div className="chief-complaint-selector">
                    {complaintOptions.map(option => (
                      <button key={option} type="button" className={`complaint-chip ${formData.chief_complaint === option ? 'active' : ''}`} onClick={() => handleComplaintSelect(option)}>{option}</button>
                    ))}
                  </div>
                </div>

                <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', marginTop: '5px'}}>
                  <div className="form-group"><label className="form-label">나이</label><input type="number" name="age" min="18" max="85" value={formData.age} onChange={handleChange} className="form-input" required /></div>
                  <div className="form-group"><label className="form-label">체온(℃)</label><input type="number" name="temperature" min="30" max="45" step="0.1" value={formData.temperature} onChange={handleChange} className="form-input" required /></div>
                  <div className="form-group"><label className="form-label">SpO2(%)</label><input type="number" name="o2sat" min="50" max="120" value={formData.o2sat} onChange={handleChange} className="form-input" required /></div>
                </div>

                <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '5px'}}>
                  <div className="form-group"><label className="form-label">수축기 혈압(SBP)</label><input type="number" name="sbp" min="50" max="260" value={formData.sbp} onChange={handleChange} className="form-input" required /></div>
                  <div className="form-group"><label className="form-label">이완기 혈압(DBP)</label><input type="number" name="dbp" min="0" max="200" value={formData.dbp} onChange={handleChange} className="form-input" required /></div>
                  <div className="form-group"><label className="form-label">심박수(HR)</label><input type="number" name="heart_rate" min="25" max="225" value={formData.heart_rate} onChange={handleChange} className="form-input" required /></div>
                  <div className="form-group"><label className="form-label">호흡수(RR)</label><input type="number" name="resp_rate" min="7" max="40" value={formData.resp_rate} onChange={handleChange} className="form-input" required /></div>
                </div>

                <div className="form-group" style={{marginTop: '5px'}}>
                  <label className="form-label">통증 점수 (NRS 0-10): <span style={{color: '#2563EB', fontWeight: 'bold'}}>{formData.pain_score}점</span></label>
                  <input type="range" name="pain_score" min="0" max="10" value={formData.pain_score} onChange={handleChange} style={{width:'100%', cursor:'pointer', accentColor: '#2563EB'}} />
                </div>

                <button type="submit" disabled={loading} className="form-submit-button" style={{marginTop: 'auto'}}>
                  {loading ? '분석 중...' : '🚀 AI Triage 가동'}
                </button>
              </form>
            </div>

            {/* [2컬럼] 실시간 대기열 */}
            <div className="dashboard-card" style={{padding: '10px'}}>
              <div className="card-title-group" style={{padding: '5px 10px', marginBottom: '10px'}}><div className="card-title">📊 실시간 Triage 대기열</div></div>
              <div style={{overflowY: 'auto', flex: 1, padding: '5px'}}>
                {activeQueue.length > 0 ? activeQueue.map((p) => {
                  const isSelected = selectedPatient?.id === p.id;
                  return (
                    <div key={p.id} onClick={() => setSelectedPatient(p)}
                         style={{
                           border: isSelected ? '2px solid #2563EB' : '1px solid #E2E8F0',
                           borderRadius: '10px', padding: '12px', marginBottom: '12px',
                           cursor: 'pointer', backgroundColor: isSelected ? '#EFF6FF' : '#FFF',
                           display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                           boxShadow: isSelected ? '0 4px 6px -1px rgba(37, 99, 235, 0.1)' : 'none',
                           transition: 'all 0.2s'
                         }}>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: '#64748B', marginBottom: '2px' }}>{p.time.split(' ')[1]} 방문 | ID.{p.id}</div>
                        <div style={{ fontWeight: '800', fontSize: '1.1rem', color: '#1E293B' }}>
                          {maskName(p.name)} <span style={{fontSize: '0.8rem', fontWeight: 'normal', color: '#64748B'}}>({p.age}세)</span>
                        </div>
                        <div style={{ fontSize: '0.85rem', color: '#DC2626', fontWeight: 600, marginTop: '4px' }}>종합 점수: {p.score}점</div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px' }}>
                        <div className={`level-badge ${getLevelClass(p.level)}`} style={{ padding: '6px 14px', fontSize: '1.2rem' }}>Lv.{p.level}</div>
                        <button onClick={(e) => handleDischarge(p.id, e)} 
                                style={{ padding: '5px 10px', backgroundColor: '#F1F5F9', border: '1px solid #CBD5E1', borderRadius: '6px', fontSize: '0.75rem', cursor: 'pointer', color: '#475569', fontWeight: 'bold' }}>
                          진료 완료(퇴실)
                        </button>
                      </div>
                    </div>
                  )
                }) : (
                  <div style={{textAlign:'center', color:'#94A3B8', fontSize:'0.9rem', marginTop: '40px'}}>현재 대기 중인 환자가 없습니다.</div>
                )}
              </div>
            </div>

            {/* [3컬럼] XAI 리포트 및 자연어 소견 텍스트 */}
            <div className="analytics-column">
              <div className="dashboard-card xai-card" style={{display: 'flex', flexDirection: 'column'}}>
                <div className="card-title-group" style={{marginBottom: '10px', paddingBottom: '5px'}}>
                  <div className="card-title">🦾 임상 판단 근거 (XAI)</div>
                </div>
                
                {selectedPatient ? (
                  <div style={{display: 'flex', flexDirection: 'column', height: '100%'}}>
                    <div style={{fontSize: '0.85rem', color: '#1E293B', fontWeight: 600, marginBottom: '10px'}}>
                      선택된 환자: {maskName(selectedPatient.name)} (ID.{selectedPatient.id})
                    </div>
                    
                    {/* 💡 핵심: height={160} 으로 고정하여 그래프 렌더링 버그 완벽 차단! */}
                    {selectedPatient.xai_data ? (
                      <div className="chart-container" style={{flex: 1, minHeight: '140px'}}>
                        <ResponsiveContainer width="100%" height={160}>
                          <BarChart data={selectedPatient.xai_data} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
                            <XAxis type="number" hide />
                            <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} style={{fontSize: '0.75rem', fontWeight: 'bold'}} width={40} />
                            <RechartsTooltip cursor={{fill: 'transparent'}} contentStyle={{fontSize:'0.8rem', padding:'5px', borderRadius: '8px'}} />
                            <ReferenceLine x={0} stroke="#CBD5E1" />
                            <Bar dataKey="value" barSize={16} radius={[0, 4, 4, 0]}>
                              {selectedPatient.xai_data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#EF4444' : '#10B981'} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (<div style={{margin:'auto', color:'#94A3B8', fontSize:'0.8rem'}}>XAI 데이터 없음</div>)}

                    {/* Rule-based NLG 텍스트 소견 */}
                    {selectedPatient.warnings && selectedPatient.warnings.length > 0 && (
                      <div style={{ marginTop: '15px', display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto', maxHeight: '130px', paddingRight: '5px' }}>
                        <div style={{fontSize: '0.75rem', color: '#64748B', fontWeight: 700}}>상세 임상 소견:</div>
                        {selectedPatient.warnings.map((warn, index) => (
                          <div key={index} className="warning-item" style={{ padding: '8px 12px', margin: 0, display: 'flex', alignItems: 'flex-start', gap: '8px', backgroundColor: '#FEF2F2', border: '1px solid #FECACA', borderRadius: '6px' }}>
                            <span style={{fontSize:'1rem', lineHeight: '1.2'}}>🚨</span>
                            <span className="warning-text" style={{fontSize: '0.85rem', color: '#991B1B', fontWeight: 600, lineHeight: '1.4'}}>{warn}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{margin:'auto', color:'#94A3B8', fontSize:'0.85rem', textAlign: 'center'}}>
                    대기열에서 환자를 클릭하면<br/>AI 판단 근거가 표시됩니다.
                  </div>
                )}
              </div>

              <div className="dashboard-card fairness-card">
                <div className="card-title-group" style={{marginBottom: '10px', paddingBottom: '5px'}}><div className="card-title">🛡️ Fairness Audit</div></div>
                <div className="fairness-metrics-grid">
                  <div className="fairness-score-card">
                    <p style={{fontSize:'0.7rem', color:'#64748B'}}>연령 대기격차</p>
                    <p className="fairness-value" style={{color: '#EA580C'}}>+12분</p>
                  </div>
                  <div className="fairness-score-card">
                    <p style={{fontSize:'0.7rem', color:'#64748B'}}>성별 편향도</p>
                    <p className="fairness-value" style={{color: '#16A34A'}}>0.91</p>
                  </div>
                </div>
                <p style={{fontSize: '0.75rem', color: '#991B1B', marginTop: '10px', backgroundColor: '#FEF2F2', padding: '8px', borderRadius: '4px'}}>
                  [Flagged] 노년층 대기 시간이 유의미하게 높음
                </p>
              </div>
            </div>
          </div>
        )}

        {/* 탭 2: 환자 목록 */}
        {activeTab === 'roster' && (
          <div className="dashboard-card">
            <div className="card-title-group"><div className="card-title">👥 누적 환자 명단 (전체 기록)</div>
            <button onClick={handleExportCSV} style={{padding: '6px 12px', backgroundColor: '#10B981', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize:'0.9rem'}}>💾 CSV 추출</button></div>
            <div style={{overflowY: 'auto', flex: 1}}>
              <table style={{width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem'}}>
                <thead style={{position: 'sticky', top: 0, backgroundColor: '#F8FAFC', zIndex: 1}}>
                  <tr style={{borderBottom: '2px solid #E2E8F0', color: '#64748B'}}>
                    <th style={{padding: '10px'}}>상태</th><th style={{padding: '10px'}}>ID</th><th style={{padding: '10px'}}>성명</th>
                    <th style={{padding: '10px'}}>나이</th><th style={{padding: '10px'}}>SpO2</th><th style={{padding: '10px'}}>예측 등급</th>
                  </tr>
                </thead>
                <tbody>
                  {patientHistory.map((p, idx) => (
                    <tr key={idx} style={{borderBottom: '1px solid #E2E8F0', opacity: p.isActive ? 1 : 0.6}}>
                      <td style={{padding: '10px'}}><span style={{padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', backgroundColor: p.isActive ? '#EFF6FF' : '#F1F5F9', color: p.isActive ? '#2563EB' : '#94A3B8'}}>{p.isActive ? '대기중' : '진료완료'}</span></td>
                      <td style={{padding: '10px', color: '#64748B'}}>#{p.id}</td><td style={{padding: '10px', fontWeight: 600}}>{maskName(p.name)}</td>
                      <td style={{padding: '10px'}}>{p.age}세</td>
                      <td style={{padding: '10px', color: p.spo2 < 95 ? '#DC2626' : '#1E293B'}}>{p.spo2}%</td>
                      <td style={{padding: '10px'}}><span style={{fontWeight: 'bold', color: p.level <= 2 ? '#DC2626' : '#16A34A'}}>Level {p.level}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 탭 3: 통계 */}
        {activeTab === 'stats' && (
          <div className="dashboard-grid-3" style={{gridTemplateColumns: '1fr 1fr'}}>
            <div className="dashboard-card">
              <div className="card-title-group"><div className="card-title">📊 누적 중증도 분포</div></div>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart><Pie data={getLevelStats()} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={5} dataKey="value">{getLevelStats().map((entry, index) => <Cell key={`cell-${index}`} fill={entry.fill} />)}</Pie><RechartsTooltip /><Legend /></PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App