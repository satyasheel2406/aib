# main.py - FastAPI Backend with Auth, Reader, Chat, Flashcard, Quiz, and Planner Agents

import os
import shutil
import sqlite3 
import time 
import uuid 
import re # CRITICAL: For text sanitization
import json # NEW: For parsing structured JSON response from Gemini
from datetime import datetime, timedelta 
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel, Field
from typing import List

# LangChain/LLM Imports - CORRECTED PATHS for Google
from langchain_google_genai import ChatGoogleGenerativeAI # <<< NEW IMPORT for Gemini
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser # Still used for schema definition
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader 
from langchain_core.documents import Document 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Your defined models - CORRECTED to absolute import
from models import StructuredNotes, TopicStructure, FlashcardDeck, Flashcard, GeneratedQuiz, QuizAnswer, QuizSubmission, QuizQuestion, QuizOption

# =======================================================
# --- CONFIGURATION & INITIALIZATION ---
# =======================================================

# --- LLM and Vector Store Configuration ---
# CRITICAL: Use the Gemini model for structured JSON output
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Highly recommended for fast, structured tasks
    temperature=0,
    # This setting forces the output to be a valid JSON string (essential for Pydantic parsing)
    model_kwargs={"response_mime_type": "application/json"}
) 
UPLOAD_DIR = "uploaded_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DB_PATH = "faiss_index"
EMBEDDINGS_MODEL = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Database (SQLite) Configuration ---
DB_FILE = "study_assistant.db"

def init_db():
    """Initializes the SQLite database tables for tracking performance."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_performance (
            id INTEGER PRIMARY KEY,
            user_email TEXT NOT NULL,
            topic TEXT NOT NULL,
            date_completed TEXT NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_topic ON quiz_performance (user_email, topic);
    """)
    conn.commit()
    conn.close()

init_db()


# --- Auth Configuration ---
SECRET_KEY = "YOUR_SUPER_SECRET_KEY" 
ALGORITHM = "HS256"
mock_users_db = {
    "test@example.com": "hashedpassword", 
    "user@study.com": "securepass"
}

# --- App Setup ---
app = FastAPI(title="AI Study Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# --- AUTH MODELS & FUNCTIONS ---
# =======================================================

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserCredentials(BaseModel):
    email: str
    password: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/signin")

def create_access_token(data: dict):
    to_encode = data.copy()
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if user_email not in mock_users_db:
        raise HTTPException(status_code=401, detail="User not found")
        
    return user_email

# =======================================================
# --- READER AGENT CORE LOGIC (SEGMENTATION & INDEXING) ---
# =======================================================

def segment_and_store_topics(document_text: str, file_name: str) -> int:
    """Uses the LLM to segment raw text into structured topics and indexes the content."""
    
    # Pydantic Output Parser is still used to GET the format instructions
    parser = PydanticOutputParser(pydantic_object=StructuredNotes) 
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            # CRITICAL: Prompt focuses on rigid output, relying on model_kwargs for JSON enforcement
            ("system", 
             "You are an expert academic assistant. Your task is to analyze the study notes "
             "and segment the content into a structured list of major topics, including definitions, "
             "key takeaways, and summaries. CRITICAL: THE ONLY OUTPUT MUST BE VALID JSON THAT CONFORMS "
             "EXACTLY TO THE REQUIRED SCHEMA. Use short, factual segments."
             "\n\nSchema to follow: {format_instructions}"
            ),
            ("user", 
             "Analyze the following study material from the file '{file_name}':\n\n--- CONTENT ---\n{document_text}"
            ),
        ]
    )
    # The chain invokes the LLM configured for JSON mime-type output
    chain = prompt_template | LLM 
    
    try:
        llm_input = document_text[:12000]
        
        # 1. Invoke the LLM
        response = chain.invoke({
            "document_text": llm_input,
            "file_name": file_name,
            "format_instructions": parser.get_format_instructions(), # Pass schema to prompt
        })

        # 2. Parse the string output (response.content for Gemini)
        json_output = json.loads(response.content)

        # 3. Validate and map to Pydantic model
        structured_data = StructuredNotes.model_validate(json_output)
        
        print(f"[LLM] Successfully structured {len(structured_data.topics)} topics.")
        
    except Exception as e:
        print(f"[LLM ERROR] Failed to parse structured data or LLM call failed: {e}")
        return 0 

    documents_to_index = []
    
    for topic in structured_data.topics:
        documents_to_index.append(Document(
            page_content=topic.summary, 
            metadata={"topic": topic.topic_name, "source": file_name, "type": "Summary"}
        ))
        
        for segment in topic.key_segments:
            documents_to_index.append(Document(
                page_content=segment.content, 
                metadata={"topic": topic.topic_name, "source": file_name, "type": segment.type}
            ))

    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
        db.add_documents(documents_to_index)
        db.save_local(DB_PATH)
        print(f"[READER AGENT] Added {len(documents_to_index)} new segments to existing DB.")
    else:
        db = FAISS.from_documents(documents_to_index, EMBEDDINGS_MODEL)
        db.save_local(DB_PATH)
        print(f"[READER AGENT] Created new DB with {len(documents_to_index)} segments.")

    return len(structured_data.topics)


def run_reader_agent(file_path: str):
    """Core file loading and LLM segmentation workflow."""
    print(f"[READER AGENT] Starting processing for {file_path}")
    file_name = os.path.basename(file_path)

    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
        
    documents = loader.load()
    full_document_text = "\n\n".join([doc.page_content for doc in documents])
    
    # CRITICAL FIX: Sanitize the text by removing citation tags and multiple spaces.
    sanitized_text = re.sub(r'\]*\]|\[cite_start\]', '', full_document_text)
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
    
    topics_count = segment_and_store_topics(sanitized_text, file_name)
    
    if topics_count == 0:
        raise HTTPException(status_code=500, detail="LLM failed to structure the document content. Try a different file.")
        
    return topics_count

# =======================================================
# --- CHAT / DOUBT AGENT CORE LOGIC (RAG) ---
# =======================================================

class ChatMessage(BaseModel):
    message: str

def setup_rag_chain():
    """Initializes the Retrieval-Augmented Generation (RAG) chain."""
    
    if not os.path.exists(DB_PATH):
        print("[RAG WARNING] FAISS index not found. Chat will be disabled until a file is uploaded.")
        return None
        
    try:
        vector_store = FAISS.load_local(DB_PATH, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[RAG ERROR] Could not load FAISS index: {e}")
        return None 
        
    retriever = vector_store.as_retriever()

    rag_prompt = ChatPromptTemplate.from_template("""
        You are the AI Study Assistant Chat Agent. Your task is to answer the user's question 
        ONLY using the context provided below from their uploaded study notes.
        
        If the answer is not found in the context, clearly state: 
        "I need more information in the uploaded notes to answer that question." 
        Do not use external knowledge.

        --- CONTEXT ---
        {context}
        
        --- QUESTION ---
        {question}
    """)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | LLM
        | StrOutputParser()
    )
    
    return rag_chain

RAG_CHAIN = setup_rag_chain()


# =======================================================
# --- FLASHCARD AGENT CORE LOGIC ---
# =======================================================

def generate_flashcards_from_memory(num_concepts: int = 5) -> List[dict]:
    """Retrieves key concepts from the vector store and generates flashcards."""
    
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="Knowledge memory (FAISS index) is empty. Please upload study materials first.")
        
    vector_store = FAISS.load_local(DB_PATH, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
    concepts = vector_store.similarity_search("key definitions and important takeaways", k=num_concepts)
    
    raw_concepts_text = "\n---\n".join([f"Topic: {doc.metadata.get('topic', 'Unknown')}\nContent: {doc.page_content}" for doc in concepts])
    
    if not raw_concepts_text:
        return []

    parser = PydanticOutputParser(pydantic_object=FlashcardDeck)

    flashcard_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are the Flashcard Agent. Your job is to take the provided study concepts "
             "and strictly transform them into a list of concise Question/Answer flashcards. "
             "The output must strictly adhere to the required JSON schema. The 'topic_tag' field must be extracted directly from the text provided."
             "\n{format_instructions}"
            ),
            ("user", 
             f"Generate {num_concepts} high-quality flashcards from the following concepts:\n\n--- CONCEPTS ---\n{raw_concepts_text}"
            ),
        ]
    )
    
    flashcard_chain = flashcard_prompt | LLM # Use the LLM directly
    
    try:
        # Rely on the model's mime-type setting to return JSON string
        response = flashcard_chain.invoke({
            "concepts_text": raw_concepts_text,
            "format_instructions": parser.get_format_instructions(),
        })

        json_output = json.loads(response.content)
        structured_deck = FlashcardDeck.model_validate(json_output)
        
        return [card.model_dump() for card in structured_deck.cards]
        
    except Exception as e:
        print(f"[LLM ERROR] Flashcard generation failed: {e}")
        return []


# =======================================================
# --- QUIZ AGENT CORE LOGIC (Adaptive & DB Recording) ---
# =======================================================

def generate_quiz_from_memory(num_questions: int = 3, user_email: str = None) -> dict:
    """
    Retrieves concepts and generates a structured MCQ quiz using the LLM.
    Includes logic to prioritize weak topics if performance data exists (adaptive).
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="Knowledge memory (FAISS index) is empty. Please upload study materials first.")
        
    vector_store = FAISS.load_local(DB_PATH, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
    
    # 1. Adaptive Logic: Identify Weakest Topic (Simplified)
    weak_topic = None
    if user_email:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT topic, AVG(CAST(score AS REAL) / total_questions) as avg_score 
            FROM quiz_performance 
            WHERE user_email = ? 
            GROUP BY topic 
            ORDER BY avg_score ASC 
            LIMIT 1
        """, (user_email,))
        
        result = cursor.fetchone()
        if result and result[1] is not None and result[1] < 0.8:
            weak_topic = result[0]
            print(f"[QUIZ AGENT] Focusing on weak topic: {weak_topic}")
        conn.close()

    # 2. Retrieve relevant concepts
    query = f"key concepts for multiple choice questions, preferably focusing on {weak_topic}" if weak_topic else "key definitions and important concepts"
    concepts = vector_store.similarity_search(query, k=num_questions)
    
    raw_concepts_text = "\n---\n".join([f"Topic: {doc.metadata.get('topic', 'Unknown')}\nContent: {doc.page_content}" for doc in concepts])
    
    if not raw_concepts_text:
        return {}

    parser = PydanticOutputParser(pydantic_object=GeneratedQuiz)

    quiz_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are the Quiz Agent. Your job is to generate a challenging multiple-choice quiz "
             f"with exactly {num_questions} questions based on the provided concepts. "
             "Each question must have 4 distinct options (ID 0, 1, 2, 3), and you must specify the correct_option_id. "
             "The options must be plausible distractors. The output must strictly adhere to the required JSON schema."
             "\nSchema to follow: {format_instructions}"
            ),
            ("user", 
             f"Generate a {num_questions}-question quiz focusing on {weak_topic or 'the general material'} from the following study material:\n\n--- CONCEPTS ---\n{raw_concepts_text}"
            ),
        ]
    )
    
    quiz_chain = quiz_prompt | LLM # Use the LLM directly
    
    try:
        quiz_id = str(uuid.uuid4())
        
        response = quiz_chain.invoke({
            "concepts_text": raw_concepts_text,
            "format_instructions": parser.get_format_instructions(),
        })

        json_output = json.loads(response.content)
        structured_quiz = GeneratedQuiz.model_validate(json_output)
        
        structured_quiz.quiz_id = quiz_id
        
        return structured_quiz.model_dump()
        
    except Exception as e:
        print(f"[LLM ERROR] Quiz generation failed: {e}")
        raise HTTPException(status_code=500, detail="Quiz generation failed due to LLM parsing error.")


def record_quiz_result(user_email: str, quiz_data: List[QuizAnswer]):
    """Records the results of a submitted quiz into the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    total_correct = 0
    total_questions = len(quiz_data)
    topic_results = {}

    for answer in quiz_data:
        topic = answer.topic
        is_correct = 1 if answer.selected_option_id == answer.correct_option_id else 0
        
        if topic not in topic_results:
            topic_results[topic] = {'correct': 0, 'total': 0}
            
        topic_results[topic]['correct'] += is_correct
        topic_results[topic]['total'] += 1
        total_correct += is_correct

    for topic, results in topic_results.items():
        cursor.execute(
            """
            INSERT INTO quiz_performance (user_email, topic, date_completed, score, total_questions) 
            VALUES (?, ?, ?, ?, ?)
            """, 
            (user_email, topic, datetime.now().isoformat(), results['correct'], results['total'])
        )
        print(f"[DB] Recorded performance for {user_email} on topic '{topic}': {results['correct']}/{results['total']}")

    conn.commit()
    conn.close()
    
    return total_correct, total_questions

# =======================================================
# --- PLANNER AGENT CORE LOGIC (Adaptive Schedule) ---
# =======================================================

def calculate_next_review(avg_score: float) -> str:
    """Calculates the next review date based on the topic's performance score (SRS logic)."""
    if avg_score is None:
        delay_days = 1
    elif avg_score < 0.5:
        delay_days = 1
    elif avg_score < 0.8:
        delay_days = 3
    else:
        delay_days = 7
        
    next_date = datetime.now() + timedelta(days=delay_days)
    return next_date.strftime('%Y-%m-%d')


def get_adaptive_review_schedule(user_email: str) -> List[dict]:
    """Queries the database to generate the next review schedule for the user."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Find the latest performance data for each topic
    cursor.execute("""
        SELECT 
            topic, 
            MAX(date_completed) as last_reviewed,
            AVG(CAST(score AS REAL) / total_questions) as avg_score
        FROM quiz_performance 
        WHERE user_email = ?
        GROUP BY topic
    """, (user_email,))
    
    performance_data = cursor.fetchall()
    conn.close()
    
    tasks = []
    
    for topic, last_reviewed, avg_score in performance_data:
        next_review_date = calculate_next_review(avg_score)
        
        tasks.append({
            "topic": topic,
            "next_review": next_review_date,
            "last_reviewed": last_reviewed,
            "avg_score_percent": round(avg_score * 100) if avg_score is not None else 0
        })
        
    return tasks


# =======================================================
# --- ENDPOINTS ---
# =======================================================

@app.post("/api/signin", response_model=Token)
async def sign_in_for_access_token(credentials: UserCredentials):
    """Handles user sign-in and issues a JWT token."""
    if credentials.email not in mock_users_db or credentials.password not in mock_users_db[credentials.email]:
        raise HTTPException(
            status_code=400,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": credentials.email})
    print(f"[AUTH] User {credentials.email} signed in and received token.")
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/signup", response_model=Token)
async def sign_up_and_issue_token(credentials: UserCredentials):
    """Handles user sign-up and issues a JWT token."""
    if credentials.email in mock_users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    mock_users_db[credentials.email] = credentials.password 
    access_token = create_access_token(data={"sub": credentials.email})
    print(f"[AUTH] New user {credentials.email} signed up and received token.")
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...), 
    current_user: str = Depends(get_current_user)
):
    """Reader Agent Endpoint: Receives file, triggers segmentation/indexing, and reloads RAG chain."""
    global RAG_CHAIN
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        topics_count = run_reader_agent(file_location)
        RAG_CHAIN = setup_rag_chain()

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Failed to process file: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    return {
        "filename": file.filename, 
        "status": "File segmented and indexed successfully", 
        "topics_added": topics_count, 
        "user": current_user
    }


@app.post("/api/chat")
async def chat_with_assistant(
    message: ChatMessage,
    current_user: str = Depends(get_current_user)
):
    """Chat/Doubt Agent Endpoint: Handles user queries using the RAG pipeline."""
    if RAG_CHAIN is None:
        raise HTTPException(status_code=503, detail="Knowledge memory is unavailable. Please upload study materials first.")
        
    try:
        print(f"[CHAT] Query from {current_user}: {message.message}")
        response_text = RAG_CHAIN.invoke(message.message)
        print(f"[CHAT] Response generated.")
        return {"response": response_text}
        
    except Exception as e:
        print(f"[ERROR] Chat RAG failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {e}")


@app.post("/api/flashcards/generate")
async def generate_flashcards_endpoint(current_user: str = Depends(get_current_user)):
    """Flashcard Agent Endpoint: Generates a deck of flashcards based on indexed notes."""
    print(f"[FLASHCARD AGENT] Generating cards for {current_user}...")
    
    try:
        flashcards = generate_flashcards_from_memory(num_concepts=5)
        
        if not flashcards:
            return {"status": "No cards generated.", "cards": []}
            
        return {"status": "success", "cards": flashcards}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Flashcard endpoint failure: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during flashcard generation.")


@app.post("/api/quizzes/generate")
async def generate_quiz_endpoint(current_user: str = Depends(get_current_user)):
    """Quiz Agent Endpoint: Generates a new quiz."""
    print(f"[QUIZ AGENT] Generating quiz for {current_user}...")
    try:
        quiz = generate_quiz_from_memory(num_questions=3, user_email=current_user)
        return quiz
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Quiz generation endpoint failure: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during quiz generation.")

@app.post("/api/quizzes/submit")
async def submit_quiz_endpoint(submission: QuizSubmission, current_user: str = Depends(get_current_user)):
    """Quiz Agent Endpoint: Receives and records student performance."""
    print(f"[QUIZ AGENT] Submitting results for quiz {submission.quiz_id} by {current_user}...")
    
    try:
        total_correct, total_questions = record_quiz_result(current_user, submission.answers)
        
        return {
            "status": "Performance recorded.",
            "score": total_correct,
            "total": total_questions,
            "message": f"You scored {total_correct} out of {total_questions}."
        }
    except Exception as e:
        print(f"[ERROR] Quiz submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to record quiz results.")


@app.get("/api/planner/tasks")
async def get_planner_tasks_endpoint(current_user: str = Depends(get_current_user)):
    """Planner Agent Endpoint: Returns the adaptive study schedule for the user."""
    print(f"[PLANNER AGENT] Generating schedule for {current_user}...")
    try:
        tasks = get_adaptive_review_schedule(current_user)
        
        tasks.sort(key=lambda x: x['next_review'])
        
        return {"tasks": tasks}
    except Exception as e:
        print(f"[ERROR] Planner endpoint failure: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate planner tasks.")
    
@app.get("/api/performance")
async def get_performance_dashboard(current_user: str = Depends(get_current_user)):
    """Returns detailed performance metrics, including overall score and weak topics."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Calculate Overall Performance
    cursor.execute("""
        SELECT 
            SUM(score), 
            SUM(total_questions)
        FROM quiz_performance 
        WHERE user_email = ?
    """, (current_user,))
    
    total_correct, total_total = cursor.fetchone()
    
    # 2. Calculate Performance per Topic
    cursor.execute("""
        SELECT 
            topic, 
            SUM(score), 
            SUM(total_questions),
            AVG(CAST(score AS REAL) * 100 / total_questions) as avg_percent
        FROM quiz_performance 
        WHERE user_email = ?
        GROUP BY topic
        ORDER BY avg_percent ASC
    """, (current_user,))
    
    topic_scores = []
    for topic, correct, total, avg_percent in cursor.fetchall():
        topic_scores.append({
            "topic": topic,
            "correct": correct,
            "total": total,
            "avg_percent": round(avg_percent, 1)
        })

    conn.close()
    
    overall_score = round((total_correct / total_total) * 100, 1) if total_total else 0
    
    return {
        "overall_score": overall_score,
        "total_quizzes": len(topic_scores),
        "total_questions_answered": total_total or 0,
        "topic_breakdown": topic_scores
    }

@app.get("/api/hello")
async def hello(current_user: str = Depends(get_current_user)):
    """A test endpoint requiring authentication."""
    return {"message": f"Hello, {current_user}! Your token works."}