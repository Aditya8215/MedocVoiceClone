import streamlit as st
import google.generativeai as genai
import json
import os

# Doctor Dictation and Doctor-Patient Conversation Script Generation
def generate_medical_script(dictation_type="Doctor Dictation"):
    try:
        with st.spinner(f"Generating new {dictation_type.lower()} script..."):
            model = genai.GenerativeModel('models/gemini-2.0-flash-lite')

            if dictation_type == "Doctor Dictation":
                system_prompt = f"""
                ##*Context*:-
                #*You are MedScribe Simulator. Your task is to generate India-specific, realistic medical dictation scripts for QA testers to read aloud when testing an AI medical scribe.*
                {st.session_state.get("is_healthcare")}='Yes'
                ##*Generate heathcare specific context in the script*
                {st.session_state.get("is_healthcare")}='No'
                ##*Generate without heathcare specific context in the script*
                ##*Guidelines*:
                - *Do not repeat script*
                - Restrcit use of punctuators other than neccessary gaps for pauses. 
                - Output only the dictation script text, no titles, no formatting, no explanations.
                - *generate not more than 100words*
                - The script must sound like spoken dictation a doctor would give, with pause neccessary.
                - Cover: patient intro, symptoms, exam findings, impression/diagnosis, treatment plan, and prescription.
                - Use Indian medical context and safe, generic prescriptions.
                - Script length: 150–450 words,1–3 minutes reading time
                -Vary specialty, severity, and dictation style across scripts.
                -Include edge cases in some scripts (unclear speech, mid-sentence correction, abrupt stop).
                -*Use more medicine or durg names*
                #*Output rule*:
                Return only the dictation script text as if spoken by the doctor.
                - *START DIRECT DICTATION- example::-- do not include-here we go,let's start direct dicatation or any such kind*
                
                """
            else:  # Doctor-Patient Conversation
                system_prompt = f"""
            You are MedScribe Simulator. Your task is to generate a realistic, India-specific, fictional doctor-patient conversation script for QA testers to read aloud.
                ##*Context*:
                #*You are MedScribe Simulator. Your task is to generate India-specific, realistic medical dictation scripts for QA testers to read aloud when testing an AI medical scribe.*
                {st.session_state.get("is_healthcare")}=='Yes'
                ##*If the healthcare industry is selected as 'Yes', then include healthcare context in the
                {st.session_state.get("is_healthcare")}=='No'
                generate a general medical dictation script without specific healthcare context.*
                ##*Guidelines*:
                - *Do not repeat script *
                - Output ONLY the diarized script text, with no titles, headers, or explanations.
                - The script must be a back-and-forth dialogue.
                - Start each line with either "Doctor: " or "Patient: ".
                - The conversation should cover patient complaints, doctor's questions, diagnosis, and a treatment plan.
                - Use Indian medical context and safe, generic prescriptions.
                - Script length should be between 150 and 450 words.

                Output rule:
                Return only the conversation script text itself. For example:
                Doctor: Good morning, how can I help you?
                Patient: I have been having a severe headache for three days, doctor.
                """
            response = model.generate_content(system_prompt)
            return response.text.strip()
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return "Failed to generate script."


def transcribe_audio_only(audio_path):
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found at {audio_path}"}
    try:
        with st.spinner('Uploading file for transcription---'):
            audio_file = genai.upload_file(path=audio_path)
        with st.spinner('Transcribing audio... This may take a moment.'):
            model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
            prompt = """
            #*IF a single speaker- Doctor dictation, transcribe as-is*.
            You are a specialized audio-to-text converter for doctor-patient conversations. Your task is to transcribe noisy phone audio recordings into clean, diarized text output with maximum speaker identification accuracy.
            ## CORE MISSION
            Convert audio input into structured dialogue format using only two speaker labels: DOCTOR: and PATIENT:

            ## AUDIO CONTEXT AWARENESS
            - Input: Noisy phone recordings with distant microphone placement
            - Expect: Background noise, potential speech overlap, varying audio quality
            - Challenge: Distinguish between two speakers in suboptimal conditions

            ## PROCESSING PROTOCOL

            ### 1. AUDIO ANALYSIS FIRST
            - Identify distinct vocal characteristics (pitch, pace, speech patterns)
            - Map higher/lower frequency ranges to likely speaker types
            - Note conversation flow patterns (questions vs responses, medical terminology usage)

            ### 2. SPEAKER IDENTIFICATION STRATEGY
            - DOCTOR typically: Uses medical terminology, asks diagnostic questions, provides instructions/explanations
            - PATIENT typically: Describes symptoms, asks clarifying questions, responds to medical queries
            - When uncertain: Use contextual clues from conversation content rather than guessing

            ### 3. TRANSCRIPTION RULES
            - **CRITICAL:** Pay extremely close attention to negations and qualifications (e.g., "not," "don't," "can't," "I'm not," "a little," "sometimes"). A missed "not" can completely invert the clinical meaning. Transcribe these words with high fidelity.
            - Format every line as either "DOCTOR: [speech]" or "PATIENT: [speech]"
            - Maintain natural conversation flow and timing
            - Include hesitations, partial words only if they affect meaning
            - Mark unclear audio as [UNCLEAR] rather than guessing
            - Use [OVERLAPPING] when both speakers talk simultaneously

            ### 4. MEDICAL CONTEXT HANDLING
            - Preserve all medical terms accurately
            - Maintain patient privacy (don't add identifying details not in audio)
            - Keep symptom descriptions verbatim
            - Preserve medication names and dosages exactly as spoken

            ### 5. QUALITY ASSURANCE
            - Cross-reference speaker assignments with conversation logic
            - Verify medical terminology context matches speaker role
            - Flag inconsistent speaker patterns with [SPEAKER_UNCERTAIN] tag
            - Prioritize accuracy over perfection - mark uncertainties rather than guess

            ## OUTPUT FORMAT

            Doctor: [First speaker's words]
            Patient: [Second speaker's words]
            Doctor: [Continuing dialogue]
            [UNCLEAR] [when audio is unintelligible]
            [OVERLAPPING] Doctor: [speech] / Patient: [speech]


            ## ERROR HANDLING
            When speaker identification confidence is low:
            - Use context clues from medical conversation patterns
            - Mark uncertainty with [SPEAKER_UNCERTAIN] before the line
            - Never leave speech unattributed - assign to most likely speaker

            EXECUTE: Process the provided audio and return clean diarized transcript following this protocol.
            """
            response = model.generate_content([prompt, audio_file], request_options={"timeout": 600})
            return response.text
    except Exception as e:
        return {"error": f"An error occurred during transcription: {e}"}


def extract_prescription_from_text(transcription_text):
    try:
        with st.spinner('Generating prescription from transcription...'):
            model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
            prompt = f"""
            #*You are a highly intelligent medical data extraction system. You are given a text transcription of a doctor-patient consultation.*
            #Your task is to extract key medical information from this text and format it into a single, valid JSON object according to the rules and structure below.
            ### TRANSCRIPTION TEXT ###
            {transcription_text}
            ### RULES & JSON STRUCTURE ###
            # JSON OUTPUT STRUCTURE
            {{
                "name": "", "date": "", "time": "", "doctorUsername": "", "patientUsername": "", "hospitalName": "", "hospitalId": "", "clinicalNote": "", "diagnosis": [], "complaints": [], "notes": [],
                "medication": [{{"name": "", "medicationDetails": [{{"dose": "","dosage": "","route": "","freq": "","dur": "","class": "","when": ""}}]}}],
                "test": [{{"name": "","instruction": "","date": ""}}], "followup": {{"date": "","reason": ""}},
                "vitals": {{"BP": "","Heartrate": "","RespiratoryRate": "","temp": "","spO2": "","weight": "","height": "","BMI": "","waist_hips": ""}},
                "nursing": [{{"instruction": "","priority": ""}}], "discharge": {{"planned_date": "","instruction": "","Home_Care": "","Recommendations": ""}},
                "icdCode": [], "medicalHistory": [], "labScanPdf": [], "systematicExamination": {{"General": [],"CVS": [],"RS": [],"CNS": [],"PA": [],"ENT": []}},
                "assessmentPlan": "", "nutritionAssessment": [],
                "referredTo": {{"doctorName": "","doctorUsername": "","phoneNumber": "","email": "","hospitalId": "","hospitalName": "","speciality": ""}},
                "scribePrescription": {{"scribeId": "","imageUrl": "","publicId": "","date": ""}}
            }}
            """
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            parsed_json = json.loads(response.text.strip())
            return parsed_json
    except Exception as e:
        return {"error": f"An error occurred while generating the prescription: {e}"}