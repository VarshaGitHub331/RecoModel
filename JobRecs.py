import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing CORS to handle cross-origin requests

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client using the new method
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the index name (this should match the one in Pinecone dashboard)
INDEX_NAME ="job-recommendation"

# Initialize the Sentence Transformer model
model = SentenceTransformer('fine_tuned_skill_model')  # Replace with your fine-tuned model

# Step 1: Create or connect to a Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # The embedding dimension size for the model used (this is for BERT-based models)
        metric='cosine',  # You can also use 'euclidean', 'dotproduct', etc.
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to the Pinecone index
index = pc.Index(INDEX_NAME)

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

@app.route("/add-job", methods=["POST"])
def add_job():
    print("HEREEEE FOR EMBEDDINGSSS")
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging line to check the received data
        job_id = data.get("job_id")
        job_skills = data.get("job_skills")

        if not job_id or not job_skills:
            return jsonify({"error": "Missing job ID or skills"}), 400

        # Generate embedding for job skills
        skills_embedding = model.encode(job_skills)
        
        # Insert or update job information into Pinecone
        # The job_id is used as the unique vector ID
        metadata = {
            "job_skills": job_skills
        }

        # Insert or update the job vector in Pinecone
        index.upsert(
            vectors=[(job_id, skills_embedding.tolist(), metadata)]
        )

        return jsonify({"message": "Job skills added or updated successfully", "job_id": job_id}), 200

    except Exception as e:
        print("Error:", e)  # Debugging line to print the error
        return jsonify({"error": str(e)}), 500

@app.route('/get_similar_jobs', methods=['POST'])
def get_similar_jobs():
    data = request.get_json()
    print("Received data:", data)

    user_skills = data.get('user_skills')
    k = int(request.json.get('k', 3))  # Assuming 3 is the default if not provided

    if not user_skills:
        return jsonify({"error": "'user_skills' is required"}), 400

    # Get the embedding of the user skills
    user_embedding = model.encode(user_skills)
    print("User embedding:", user_embedding)  # Debugging the generated user embedding

    # Perform the query on Pinecone
    results = index.query(vector=user_embedding.tolist(), top_k=k)

    # Process the results
    recommended_jobs = []
    for match in results['matches']:
        job_id = match['id']
        score = match['score']
        recommended_jobs.append({"job_id": job_id})

    return jsonify({"recommended_jobs": recommended_jobs})

if __name__ == "__main__":
    app.run(debug=True)
