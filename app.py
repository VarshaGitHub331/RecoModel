from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Fine-Tuned Model
fine_tuned_model = SentenceTransformer('fine_tuned_skill_model')  # Ensure the model is loaded correctly

# Example Skill Matching: Use the Fine-Tuned Model to Extract Relevant Skills
db_skills = [
    "Python", "Machine Learning", "Java", "Data Analysis", 
    "Deep Learning", "Artificial Intelligence", "React", 
    "SQL", "Node.js", "MongoDB", "JavaScript", "HTML", "CSS"
]

generated_skills = ["Python", "Artificial Intelligence", "SQL", "Machine Learning", "React"]

# Encode the Skills into Embeddings
db_embeddings = fine_tuned_model.encode(db_skills)
gen_embeddings = fine_tuned_model.encode(generated_skills)

# Calculate Cosine Similarity between generated skills and database skills
similarity = cosine_similarity(gen_embeddings, db_embeddings)

# Extract the Most Relevant Skill for each generated skill
matched_skills = []
for i, gen_skill in enumerate(generated_skills):
    matched_index = similarity[i].argmax()  # Find the most similar skill from db_skills
    matched_skills.append({
        "generated_skill": gen_skill,
        "most_similar_skill": db_skills[matched_index],
        "similarity_score": similarity[i][matched_index]
    })

# Print the Matched Skills and their Similarity Scores
for match in matched_skills:
    print(f"Generated Skill: {match['generated_skill']} -> Most Similar Skill: {match['most_similar_skill']} (Score: {match['similarity_score']:.2f})")

# Below code (Flask-related) is commented out as it's not needed for this direct testing

from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

@app.route("/extract_skills", methods=["POST"])
def extract_skills():
    try:
        # Parse JSON request data
        data = request.get_json()
        db_skills = data.get("db_skills", [])
        generated_skills = data.get("generated_skills", [])
        
        # Validate input
        if not db_skills or not generated_skills:
            return jsonify({"error": "Both 'db_skills' and 'generated_skills' must be provided as non-empty lists."}), 400
        
        # Generate embeddings for database and generated skills
        db_embeddings = fine_tuned_model.encode(db_skills)
        generated_embeddings = fine_tuned_model.encode(generated_skills)
        mostSimilarSkill = ""
        prevMatched = -1
        # Find the most similar skill for each generated skill
        extracted_skills = []
        for embed in generated_embeddings:
            similarity = cosine_similarity([embed], db_embeddings)  # similarity is 2D (1, n)
            index = similarity[0].argmax()
              # Access the first row
            if similarity[0][index] > prevMatched:
                prevMatched = similarity[0][index]
                mostSimilarSkill = db_skills[index]
            extracted_skill = db_skills[index]
            extracted_skills.append(extracted_skill)

        # Return the extracted skills as JSON
        return jsonify({"extracted_skills": extracted_skills, "most_similar_skill": mostSimilarSkill}), 200

    except Exception as e:
        # Log and handle errors gracefully
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

