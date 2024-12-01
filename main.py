from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
from vertex_lib import *
import json
import pandas as pd
from datetime import datetime
from typing import List, Optional

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('question_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

success_logger = logging.getLogger('success_logger')
success_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('correct.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
success_logger.addHandler(file_handler)

# Pydantic models
class ContentSection(BaseModel):
    section_title: str
    page_numbers: List[int]
    relevance_score: float

class QuestionRequest(BaseModel):
    course_content_urls: List[str]  # Accept multiple URLs
    num_questions: Optional[int] = Field(default=3, ge=1, le=10)

class Question(BaseModel):
    question_text: str
    learning_outcomes: List[str]  
    program_competencies: List[str]  
    grading_guidelines: str
    relevant_sections: List[ContentSection]
    estimated_time_minutes: int

class QuestionsResponse(BaseModel):
    timestamp: str
    course_urls: List[str]
    questions: List[Question]

class QuestionGenerator:
    def __init__(self):
        self.logger = logger
        self.model_arch = ChatMainLLM.VERTEXLLM.value
        self.model_name = ChatMainLLMName.VERTEX_GEMINI_15_PRO_2
        self._initialize_llm()
        self._load_course_data()
        
    def _initialize_llm(self):
        try:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    self.curl_vertex = CurlVertex(
                        logger=self.logger,
                        llm_config=LLMConfig({
                            "model_arch": self.model_arch,
                            "model_name": self.model_name,
                            "temperature": 0.7,
                            "top_k": 40,
                            "top_p": 0.95,
                            "max_tokens": 2048
                        })
                    )
                    # Remove or comment out the test_connection call since it's not implemented
                    # self.curl_vertex.test_connection()
                    self.logger.info("CurlVertex initialized successfully")
                    break
                    
                except Exception as e:
                    if "ACCESS_TOKEN_EXPIRED" in str(e) or "UNAUTHENTICATED" in str(e):
                        self.logger.warning(f"Authentication token expired, attempting refresh (attempt {retry_count + 1}/{max_retries})")
                        # Force token refresh
                        self.curl_vertex = None
                        retry_count += 1
                        if retry_count == max_retries:
                            raise Exception("Failed to refresh authentication token after maximum retries")
                    else:
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error initializing CurlVertex: {str(e)}")
            raise
    def _get_llm_response(self, chat_history: List[Message], instruction_prompt: str) -> str:
        """Gets response from LLM and cleans markdown formatting."""
        self.logger.debug("Sending request to CurlVertex")
        full_response = ""
        
        try:
            for response in self.curl_vertex.generate(
                instruction_prompt=instruction_prompt,
                chat_history=chat_history,
                is_streaming=True,
                return_tokens=True,
                timeout=500
            ):
                full_response += response.content
            
            # Log raw response
            self.logger.debug(f"Raw response: {full_response}")
            
            # Clean the response
            cleaned_response = self._clean_json_response(full_response)
            
            # Validate JSON
            try:
                json.loads(cleaned_response)  # Test if it's valid JSON
                return cleaned_response
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON response: {e}")
                self.logger.debug(f"Attempted to parse: {cleaned_response}")
                raise ValueError(f"Failed to parse LLM response as JSON: {e}")
                
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            raise

    def _clean_json_response(self, response: str) -> str:
        """Cleans and validates JSON response from LLM."""
        cleaned = response.strip()
        
        # Remove JSON code block markers
        if cleaned.startswith("```json\n"):
            cleaned = cleaned[8:]
        elif cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("\n```"):
            cleaned = cleaned[:-4]
        elif cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Fix common JSON formatting issues
        cleaned = cleaned.replace('\n', ' ')  # Replace newlines with spaces
        cleaned = ' '.join(cleaned.split())   # Normalize whitespace
        cleaned = cleaned.replace('} {', '}, {')  # Fix missing commas between objects
        cleaned = cleaned.replace(']"', '], "')   # Fix missing commas after arrays
        cleaned = cleaned.replace('}"', '}, "')   # Fix missing commas after objects
        
        # Log cleaned response
        self.logger.debug(f"Cleaned response: {cleaned}")
        success_logger.info(f"Cleaned LLM Response: {cleaned}")
        
        return cleaned

    def _load_course_data(self):
        try:
            self.learning_outcomes = pd.read_csv("oc.csv")
            self.program_competencies = pd.read_csv("py.csv")
            
            # Learning outcomes mapping
            self.lo_mapping = {
                f"Ö{row['Sayı']}": row['Öğrenme Çıktısı']
                for _, row in self.learning_outcomes.iterrows()
            }
            
            # Program competencies mapping
            self.pc_mapping = {
                int(row['Sıra No']): row['Program Yeterlikleri']
                for _, row in self.program_competencies.iterrows()
            }
            
            # Rest of the method remains the same
            self.competencies_data = {
                "program_competencies": [],
                "learning_outcomes": self.lo_mapping
            }
            
            for _, row in self.program_competencies.iterrows():
                competency = {
                    "id": int(row['Sıra No']),
                    "description": row['Program Yeterlikleri'],
                    "learning_outcomes": []
                }
                
                for i in range(1, 5):
                    lo_key = f'Ö{i}'
                    competency["learning_outcomes"].append({
                        "id": lo_key,
                        "relevance_score": int(row[lo_key]) if pd.notna(row[lo_key]) else 0
                    })
                
                self.competencies_data["program_competencies"].append(competency)
            
            self.logger.info("Course data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading course data: {str(e)}")
            raise

    def _create_instruction_prompt(self, num_questions: int, content_url: str) -> str:
        return f"""Generate {num_questions} university-level open-ended questions in Turkish based on these competencies and learning outcomes:

{json.dumps(self.competencies_data, indent=2, ensure_ascii=False)}
Each question must:
        1. Connect to specific learning outcomes (1 to 4) from oc.csv and program competencies (1 to 15) from py.csv
        Ö1 Ö2 Ö3 Ö4 is the specific learning outcome
        2. Include detailed grading guidelines
        3. Reference relevant sections from the course material

Return in this JSON format:
{{
    "questions": [
        {{
            "question_text": "Question text",
            "competencies": [
                {{
                    "competency_id": 1,
                    "learning_outcomes": [
                        {{
                            "id": "Ö1",
                            "relevance_score": 5,
                            "id": "Ö2",
                            "relevance_score": 5
                            "id": "Ö3",
                            "relevance_score": 5
                            "id": "Ö4",
                            "relevance_score": 5
                        }}
                    ]
                }}
            ],
            "grading_guidelines": "Guidelines",
            "relevant_sections": [
                {{
                    "section_title": "Section name",
                    "page_numbers": [1, 2],
                    "relevance_score": 0.95
                }}
            ],
            "estimated_time_minutes": 30
        }}
    ]
}}"""
    def _validate_and_fix_response(self, response_data: dict) -> dict:
        """Validates and fixes the response data"""
        try:
            if not isinstance(response_data, dict) or "questions" not in response_data:
                raise ValueError("Invalid response format")
                
            for question in response_data["questions"]:
                if not isinstance(question.get("competencies"), list):
                    raise ValueError("Invalid competencies format")
                    
                for competency in question["competencies"]:
                    comp_id = competency.get("competency_id")
                    if comp_id is None or comp_id not in self.pc_mapping:
                        raise ValueError(f"Invalid competency_id: {comp_id}")
                    
                    # Get learning outcomes from program_competencies DataFrame
                    comp_row = self.program_competencies[
                        self.program_competencies['Sıra No'] == comp_id
                    ].iloc[0]
                    
                    # Update learning outcomes with actual values from CSV
                    competency["learning_outcomes"] = []
                    for i in range(1, 5):
                        lo_key = f'Ö{i}'
                        score = int(comp_row[lo_key]) if pd.notna(comp_row[lo_key]) else 0
                        if score > 0:  # Only include relevant learning outcomes
                            competency["learning_outcomes"].append({
                                "id": lo_key,
                                "relevance_score": score
                            })
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error in validate_and_fix_response: {str(e)}")
            raise ValueError(f"Failed to validate response: {str(e)}")
    def _process_url(self, url: str, num_questions: int) -> dict:
        """Process a single URL to generate questions."""
        try:
            instruction_prompt = self._create_instruction_prompt(num_questions, url)
            chat_history = [
                Message(
                    role="user",
                    message=instruction_prompt,
                    message_content_type=ChatInputContentType.PDF,
                    message_uri=url,
                    message_type=None
                )
            ]
            # Get LLM response
            full_response = self._get_llm_response(chat_history, instruction_prompt)
            response_data = json.loads(full_response)
            validated_response = self._validate_and_fix_response(response_data)
            return validated_response
        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {e}")
            return None

    def generate_questions(self, course_content_urls: List[str], num_questions: int = 3) -> List[Question]:
        self.logger.info(f"Generating questions for URLs: {course_content_urls}")

        results = []
        with ThreadPoolExecutor(max_workers=len(course_content_urls)) as executor:
            future_to_url = {executor.submit(self._process_url, url, num_questions): url for url in course_content_urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in future for URL {url}: {e}")

        if not results:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate questions from all URLs"
            )

        all_questions = []
        for result in results:
            all_questions.extend(self._parse_llm_response(result))
        
        return all_questions


    def _parse_llm_response(self, response_data: dict) -> List[Question]:
        """
        Parses the LLM response and converts it into a list of Question objects.
        
        Args:
            response_data (dict): The validated response data from the LLM
            
        Returns:
            List[Question]: A list of parsed Question objects
            
        Raises:
            HTTPException: If parsing fails
        """
        try:
            if not isinstance(response_data, dict) or "questions" not in response_data:
                raise ValueError("Invalid response format: missing 'questions' key")
                
            questions = []
            for q_data in response_data["questions"]:
                # Initialize collections for competencies and learning outcomes
                learning_outcomes = set()  # Using set to avoid duplicates
                program_competencies = set()  # Using set to avoid duplicates
                
                # Validate competencies data
                if not isinstance(q_data.get("competencies"), list):
                    raise ValueError(f"Invalid competencies format in question: {q_data}")
                    
                # Process each competency
                for comp in q_data["competencies"]:
                    comp_id = comp.get("competency_id")
                    if comp_id is None:
                        raise ValueError(f"Missing competency_id in: {comp}")
                        
                    # Get competency description from DataFrame
                    competency_row = self.program_competencies[
                        self.program_competencies['Sıra No'] == comp_id
                    ]
                    if competency_row.empty:
                        raise ValueError(f"Invalid competency_id: {comp_id}")
                        
                    program_competencies.add(
                        competency_row['Program Yeterlikleri'].iloc[0]
                    )
                    
                    # Process learning outcomes for this competency
                    if not isinstance(comp.get("learning_outcomes"), list):
                        raise ValueError(f"Invalid learning outcomes format in competency: {comp}")
                        
                    for lo in comp["learning_outcomes"]:
                        lo_id = lo.get("id")
                        relevance_score = lo.get("relevance_score", 0)
                        
                        if lo_id is None:
                            raise ValueError(f"Missing learning outcome id in: {lo}")
                            
                        if relevance_score > 0 and lo_id in self.lo_mapping:
                            learning_outcomes.add(self.lo_mapping[lo_id])
                
                # Validate required question fields
                required_fields = {
                    "question_text": str,
                    "grading_guidelines": str,
                    "relevant_sections": list,
                    "estimated_time_minutes": int
                }
                
                for field, expected_type in required_fields.items():
                    if field not in q_data:
                        raise ValueError(f"Missing required field: {field}")
                    if not isinstance(q_data[field], expected_type):
                        raise ValueError(f"Invalid type for {field}: expected {expected_type}")
                
                # Create and validate ContentSection objects
                relevant_sections = []
                for section in q_data["relevant_sections"]:
                    try:
                        relevant_sections.append(ContentSection(
                            section_title=section["section_title"],
                            page_numbers=section["page_numbers"],
                            relevance_score=section["relevance_score"]
                        ))
                    except Exception as e:
                        raise ValueError(f"Invalid section data: {section}. Error: {str(e)}")
                
                # Create Question object with validated data
                question = Question(
                    question_text=q_data["question_text"],
                    learning_outcomes=list(learning_outcomes),
                    program_competencies=list(program_competencies),
                    grading_guidelines=q_data["grading_guidelines"],
                    relevant_sections=relevant_sections,
                    estimated_time_minutes=q_data["estimated_time_minutes"]
                )
                questions.append(question)
                
            if not questions:
                raise ValueError("No valid questions were parsed from the response")
                
            return questions
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM response: {str(e)}"
            )

app = FastAPI(title="Question Generator API", version="1.0.0")

# Initialize generator
try:
    question_generator = QuestionGenerator()
    logger.info("QuestionGenerator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize QuestionGenerator: {e}")
    raise

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_name": question_generator.model_name,
        "learning_outcomes_count": len(question_generator.lo_mapping),
        "program_competencies_count": len(question_generator.pc_mapping)
    }

@app.post("/generate-questions", response_model=QuestionsResponse)
async def generate_questions(request: QuestionRequest):
    logger.info(f"Received request: {request}")

    try:
        questions = question_generator.generate_questions(
            request.course_content_urls,
            request.num_questions
        )

        response = QuestionsResponse(
            timestamp=datetime.now().isoformat(),
            course_urls=request.course_content_urls,
            questions=questions
        )

        # Log successful responses to correct.log
        success_logger = logging.getLogger('success_logger')
        success_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('correct.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - Generated questions: timestamp=%(timestamp)s course_urls=%(urls)s questions=%(questions)s'))
        success_logger.addHandler(file_handler)
        
        # Log with custom format
        success_logger.info('', extra={
            'timestamp': response.timestamp,
            'urls': response.course_urls,
            'questions': response.questions
        })

        logger.info("Successfully generated questions")
        return response
    except Exception as e:
        logger.error(f"Error in generate_questions endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)