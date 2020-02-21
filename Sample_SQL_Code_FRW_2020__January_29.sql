-------------------------------------------------------------------------------------------------------------------                                                            ---
--- Programmer: Felicia R. Williams M.S., Data Scientist Intitutional Research and Planning                     ---
--- Date: January 29, 2020                                                                                      ---
--- Puprose: XXXXXXXX                                                                                           ---
--- Requestor: XXXXXXXX                                                                                         ---       
-------------------------------------------------------------------------------------------------------------------
SELECT * FROM
(SELECT DISTINCT C.PERSONID "GTID", COUNT(DISTINCT DOORACCESSPOSTDATETIME) "No. Center Visits"
 FROM Main.STG_CMP_Center_TURNSTILEDATA_TBL C
 WHERE DOORACCESSPOSTDATETIME BETWEEN TO_Date('2019-08-01' ,'YYYY-MM-DD') AND TO_Date('2019-12-31' ,'YYYY-MM-DD')
 GROUP BY C.PERSONID) Center

INNER JOIN  

(SELECT DISTINCT
PROFILE.ID,
PROFILE.GENDER_CODE "Sex",
PROFILE.RACE_ETHN_COMB_NAME "Race/Ethnicity",
PROFILE.RESIDENCY_STATE_CODE "State of Residence",
PROFILE.CITZ_NAME "Citizenship",
GPA.GPA,
GPA.QLTY_PTS "No. of Credits",
TRM.LVL_NAME "Education Level",
TRM.FRSHMN_CHRT_TRM_CODE "Freshman Cohort Term",
TRM.RESIDENCY_NAME "Residency",
TRM.FTPT_NAME "Full-Time/Part-Time",
TRM.MATRICULATION_TRM_CODE "Matriculation Term",
COLLEGE.COLLEGE_NAME "College",
CLASS_LEVEL.CLS_NAME "Class",
MAX(OUTCOME.GRAD_DT_KEY) AS "GRADUATION_DATE"


FROM Main2.FACT_STU_ENRL_SNAPSHOT_TBL ENRL 
INNER JOIN Main2.DIM_STU_TRM_IND_TBL TRM ON TRM.ALTID = ENRL.ALTID AND TRM.TRM_CODE = ENRL.TRM_CODE
INNER JOIN Main2.DIM_STU_PROFILE_TBL STU_PROFILE ON ENRL.ALTID = STU_PROFILE.ALTID
INNER JOIN Main2.DIM_ACAD_COLLEGE_TBL COLLEGE ON COLLEGE.COLLEGE_KEY = ENRL.PRI_COLLEGE_KEY
INNER JOIN Main2.DIM_ACAD_CLS_TBL CLASS_LEVEL ON CLASS_LEVEL.CLS_KEY = ENRL.CLS_KEY
LEFT JOIN Main2.FACT_STU_GPA_TRM_TBL GPA ON GPA.ALTID = STU_PROFILE.ALTID AND GPA.TRM_CODE = ENRL.TRM_CODE
LEFT JOIN Main2.FACT_STU_OUTCOME_TBL OUTCOME ON OUTCOME.ALTID = ENRL.ALTID
WHERE ENRL.TRM_CODE = '201908' AND STU_PROFILE.ROW_CURR_IND = 'Y'

GROUP BY 
PROFILE.ID,
PROFILE.GENDER_CODE,
PROFILE.RACE_ETHN_COMB_NAME,
PROFILE.RESIDENCY_STATE_CODE,
PROFILE.CITZ_NAME,
GPA.GPA,
GPA.QLTY_PTS,
TRM.LVL_NAME,
TRM.FRSHMN_CHRT_TRM_CODE,
TRM.RESIDENCY_NAME,
TRM.FTPT_NAME,
TRM.MATRICULATION_TRM_CODE,
COLLEGE.COLLEGE_NAME,
CLASS_LEVEL.CLS_NAME) STUDENTS

ON STUDENTS.ID = Center.ID