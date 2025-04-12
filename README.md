# Intelligent Chatbot for Aviation Insights

This project presents an end-to-end solution for a cloud-based chatbot that can handle document-based and predictive queries. It was developed as part of a student competition focused on aviation data, with an emphasis on innovation, AI, and sustainability.

## ✈️ Project Goals

- Build a document-aware chatbot that can answer user queries like “What’s on page 10?”
- Use machine learning to predict aviation trends (e.g., flight counts, emissions)
- Generate dashboards to visualize insights (e.g., fuel consumption, seasonal trends)
- Deploy the solution on AWS cloud infrastructure

## 🏆 Competition Context

> “Sustainability in aviation isn't just a challenge—it's an opportunity to innovate.”

Over the last week, our team *Innovate Y* represented **Naveen Jindal School of Management, UT Dallas**, at the **9th Annual Business Analytics Symposium** hosted by *The University of Texas at Arlington - College of Business*. Competing against 40+ teams from 7 universities, we secured **3rd place**! 🎉

### The Challenge (Sponsored by TripAI Technologies):
Accelerate sustainability in the aviation industry using analytics and cloud technologies.

### Our Solution:
- 🚀 **Interactive Aviation Assistant**: A chatbot powered by **LangChain** and **Retrieval-Augmented Generation (RAG)** that helps optimize fuel usage and reduce CO₂ emissions through context-aware analytics.
- ☁️ **Cloud Deployment Strategy**: Designed a scalable **AWS architecture** to showcase the potential for real-world productization.

## 📂 Key Files

- `Chatbot_code.py`: Lambda handler for routing and inference
- `Feature_Selection.ipynb`: Feature engineering and model building
- `UTA_EDA.ipynb`: Exploratory Data Analysis of aviation data
- `Student Competition - Problem Statement.docx`: Problem description from the organizers

## 🧠 Technologies Used

- AWS: S3, Lex, Lambda, Kendra, RDS, SageMaker, Glue, QuickSight
- Python: Pandas, NumPy, Sklearn
- Jupyter Notebooks: Data analysis, model training
- Generative AI: LangChain and Retrieval-Augmented Generation (RAG)

## 🚀 Getting Started

1. Clone the repo and open notebooks in Jupyter:
```bash
git clone https://github.com/yourusername/aws-aviation-chatbot.git
cd aws-aviation-chatbot
jupyter notebook
```

2. Run `Feature_Selection.ipynb` to train and evaluate the ML model.

3. Deploy `Chatbot_code.py` in AWS Lambda and connect it to Amazon Lex.

## 📊 Outputs

- Trend visualizations (flights across holidays/seasons)
- Heatmaps of emissions and fuel burn
- Dashboard via Amazon QuickSight
- Chatbot handling both predictive and document-based queries

## ⚖️ License

This repository is released under the MIT License.

## 🙌 Acknowledgements

- University Collaboration on Aviation AI Challenge
- AWS Education Credits Program
