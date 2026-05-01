import json
import time
import logging
from datetime import datetime
from main import chain, grader_chain, editor_chain, retriever 

# Setup logging to track the thought process of the multi-agent system
# This log file acts as a trace for explainability
log_filename = f'agent_trace_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def run_evaluation():
    """
    Executes the evaluation suite to test RAG integrity.
    It simulates a user-AI interaction and logs the multi-agent reasoning loop.
    """
    try:
        with open('test_cases.json', 'r') as f:
            tests = json.load(f)
    except FileNotFoundError:
        print("Error: test_cases.json not found.")
        return

    results = []
    print(f" Starting Evaluation on {len(tests)} test cases.")
    print(f" Logs are saved to: {log_filename}\n")

    for test in tests:
        start_time = time.time()
        print(f"Testing ID {test['id']} [{test['category']}]: {test['question']}")
        logging.info(f"TEST CASE {test['id']} | CATEGORY: {test['category']} ---")
        logging.info(f"QUESTION: {test['question']}")
        
        # 1) Context Retrieval
        # Grounding the generation in the provided dataset
        docs = retriever.invoke(test['question'])
        context = "\n\n".join([d.page_content for d in docs])
        
        # 2) Initial Generation - The Librarian Agent
        attempts = 0
        response = chain.invoke({"reviews": context, "question": test['question']})
        logging.info(f"Initial Draft from Librarian: {response}")
        
        is_correct = False
        # 3) Self-Correction Loop - Grader and Editor Agents
        while attempts < 3:
            attempts += 1
            # The Grader acts as the 'Auditor' checking for hallucinations or logic leaps
            check = grader_chain.invoke({"context": context, "answer": response})
            logging.info(f"Attempt {attempts} - Grader's Feedback: {check}")
            
            if "CORRECT" in check.upper():
                is_correct = True
                logging.info(f"Status: PASSED at attempt {attempts}")
                break
            else:
                # The Editor refines the answer based on the Grader's feedback
                response = editor_chain.invoke({
                    "context": context, 
                    "answer": response, 
                    "feedback": check
                })
                logging.info(f"Attempt {attempts} - Editor's Refinement: {response}")
        
        duration = time.time() - start_time
        results.append({
            "id": test['id'],
            "question": test['question'],
            "final_answer": response,
            "attempts": attempts,
            "status": "PASS" if is_correct else "FAIL",
            "time": f"{duration:.2f}s"
        })

    # Generate the final performance report
    with open('REPORT.md', 'w') as f:
        f.write("# Evaluation Report: RAG Integrity Performance\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| ID | Question | Status | Attempts | Time |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['id']} | {r['question']} | {r['status']} | {r['attempts']} | {r['time']} |\n")
    
    print(f"\n Evaluation Complete! Results saved at REPORT.md")
    print(f"Check {log_filename} for the full reasoning trace.")

if __name__ == "__main__":
    run_evaluation()