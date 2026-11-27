<h1 align="center">Sentiment Analysis Using Traditional Machine Learning - LAB 01</h1>

<p align="center">
  <strong>PUSL 3205 – Large Language Models and Generative AI</strong>
</p>

<p align="center">
  <em>Text Preprocessing, BOW , TF-IDF , Logistic Regression , SVM</em>
</p>

<hr>
<h2>Practical Resources</h2>

<ul>
  <li>
    <a href="./[Lab_01_Sentiment_Analysis.ipynb](https://github.com/Sandewni-Perera/Portfolio-for-LLM-practicals/blob/main/Lab-01-Sentiment-Analysis/sentiment%20analysis.ipynb)">
      View Jupyter Notebook
    </a>
  </li>
  <li>
    <a href="./https://github.com/Sandewni-Perera/Portfolio-for-LLM-practicals/tree/main/Lab-01-Sentiment-Analysis/dataset">
      View Dataset Folder
    </a>
  </li>
</ul>
<h2 align="left">1. Introduction</h2>
This laboratory exercise examined the use of conventional machine learning methods for sentiment analysis through Python. The task included preprocessing text data, identifying significant features, and utilizing classification algorithms to assess if reviews conveyed positive or negative sentiments.
The UCI Sentiment Labelled Sentences dataset was used for the experiments, comprising reviews collected from various sources.

<h2 align="left">2. Objectives</h2>
The primary goals of this practical were to:
Utilize basic text preprocessing techniques on textual content.
Examine word usage trends in favorable and unfavorable evaluations.
Convert text into numerical representations utilizing Bag of Words (BOW) and TF-IDF methods.
Create machine learning models for predicting sentiment.
Evaluate model performance through Accuracy, Precision, Recall, and F1-score.
Analyze how trained models categorize new, unencountered sentences.

<h2 align="left">3. Dataset</h2>
The data source chosen for this experiment was the UCI Sentiment Labelled Sentences dataset. It includes categorized feedback gathered from three platforms:
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Description</th>
      <th align="center">Samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Amazon</td>
      <td>Product-related reviews</td>
      <td align="center">1,000</td>
    </tr>
    <tr>
      <td>IMDb</td>
      <td>Movie reviews</td>
      <td align="center">1,000</td>
    </tr>
    <tr>
      <td>Yelp</td>
      <td>Restaurant and service reviews</td>
      <td align="center">1,000</td>
    </tr>
    <tr>
      <td><strong>Total</strong></td>
      <td><strong>Combined dataset</strong></td>
      <td align="center"><strong>3,000</strong></td>
    </tr>
  </tbody>
</table>
Every review receives a binary sentiment label, with 1 indicating positive sentiment and 0 indicating negative sentiment.
The three separate datasets were combined to form a singular dataset for the next preprocessing and analysis phases.

<h2 align="left">4. Text preprocessing</h2>
A set of text-cleaning processes was executed to enhance the uniformity of the textual data prior to feature extraction. The steps for preprocessing included:
<ul>  
<li>Transforming every piece of text to lowercase.</li>  
<li>Eliminating punctuation and any undesirable characters.</li> 
<li>Removing unnecessary whitespace.</li>
<li>Creating an additional column that includes the refined sentences.</li>
</ul>
A comparison was made between the original sentences and their cleaned versions to analyze the alterations made during preprocessing.

<h2 align="left">5. train test split</h2>
The dataset was divided into training and testing subsets to aid in model creation and evaluation of performance.
<table>
  <thead>
    <tr>
      <th>Subset</th>
      <th align="center">Proportion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Training data</td>
      <td align="center">80%</td>
    </tr>
    <tr>
      <td>Testing data</td>
      <td align="center">20%</td>
    </tr>
  </tbody>
</table>
Stratified sampling was used to maintain the ratios of positive and negative labels in each subset. A random state of 42 was utilized to guarantee that the data division could be replicated.

<h2 align="left">7. Feature </h2>
<h3 align="left">7.1 BAG OF WORDS (BOW)</h3>
The Bag of Words model was implemented with CountVectorizer from Scikit-learn. This approach illustrated each sentence as a numerical vector derived from the frequency counts of words in the vocabulary.
The vocabulary was developed from the training data, and the resulting representation was then utilized on the test data using the identical vectorizer.

<h3 align="left">7.2 TF-IDF</h3>
The Term Frequency–Inverse Document Frequency (TF-IDF) method was applied as a supplementary feature extraction experiment. It allocates numerical values to terms based on how often they appear in a document and how they are spread throughout the set of documents.
This method was examined as a supplementary experiment since TF-IDF was part of the topic outline in the practical.

<h2 align="left">8. Model training </h2>
Three machine learning configurations were created to examine the effectiveness of various feature representations and classification methods.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Feature Representation</th>
      <th>Classification Algorithm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Model 1</td>
      <td>Bag of Words</td>
      <td>Logistic Regression</td>
    </tr>
    <tr>
      <td>Model 2</td>
      <td>TF-IDF</td>
      <td>Logistic Regression</td>
    </tr>
    <tr>
      <td>Model 3</td>
      <td>TF-IDF</td>
      <td>Support Vector Machine (SVM)</td>
    </tr>
  </tbody>
</table>
Logistic Regression and Linear Support Vector Classification (LinearSVC) were employed to identify patterns in the training data and categorize reviews as positive or negative.

<h2 align="left">9. Model evaluation</h2>
The evaluated classifiers were analyzed using four frequently utilized classification metrics: Accuracy, Precision, Recall, and F1-score.
<h3 align="left">9. Performance comparison</h3>
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th align="right">BOW + Logistic Regression</th>
      <th align="right">TF-IDF + Logistic Regression</th>
      <th align="right">TF-IDF + SVM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy</td>
      <td align="right">0.8109</td>
      <td align="right">0.8145</td>
      <td align="right"><strong>0.8255</strong></td>
    </tr>
    <tr>
      <td>Precision</td>
      <td align="right"><strong>0.8240</strong></td>
      <td align="right"><strong>0.8327</strong></td>
      <td align="right">0.8221</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td align="right">0.7942</td>
      <td align="right">0.7906</td>
      <td align="right"><strong>0.8339</strong></td>
    </tr>
    <tr>
      <td>F1-score</td>
      <td align="right">0.8088</td>
      <td align="right">0.8111</td>
      <td align="right"><strong>0.8280</strong></td>
    </tr>
  </tbody>
</table>
According to the evaluation results, the combination of TF-IDF and SVM attained the best Accuracy, Recall, and F1-score among the three configurations tested. In contrast, the combination of TF-IDF and Logistic Regression yielded the best Precision.
A grouped bar chart was created to visually contrast the evaluation metrics of the three classifiers.

<h2 align="left">10. Testing on new sentences </h2>
To investigate the behavior of the trained classifier outside the original dataset, five sentences composed by hand were added as extra test inputs.
The sentences underwent the same preprocessing function and TF-IDF transformation that were utilized during the model’s development. The trained SVM classifier was subsequently utilized to produce sentiment predictions for these inputs.
Two out of the five sentences were categorized based on the sentiment anticipated by humans. The leftover predictions showed possible challenges in understanding contextual phrases, negation, and differing viewpoints.
This experiment offered another illustration of the model's performance on new sentences and was distinct from the official test-set assessment.

<h2 align="left">11. Observation</h2>
The exercise emphasized the importance of text preprocessing and feature representation in creating sentiment classifiers based on machine learning. Transforming text data into numerical characteristics with BOW and TF-IDF allowed the classification algorithms to identify patterns linked to sentiment labels.
Of the three configurations, SVM with TF-IDF showed the best overall performance regarding Accuracy and F1-score. Still, the forecasts produced for sentences crafted manually indicated that achieving good results on the initial test set does not automatically ensure precise understanding of all unknown sentences.
The results also highlighted certain drawbacks of conventional machine learning methods for text classification, especially when sentences include contextual subtleties, negation, or more intricate expressions of viewpoint.

<h2 align="left">12. Learning outcomes</h2>
The practical provided hands on  experience with the subsequent concepts and methods:
<ul>
<li>Utilizing basic procedures for cleaning text in NLP.</li>
<li>Analyzing sentiment-tagged text via descriptive analysis.</li>
<li>Analyzing word occurrences through data visual representations.</li>
<li>Converting text into numerical formats through BOW and TF-IDF.</li>
<li>Executing Logistic Regression and SVM classifiers.</li>
<li>Assessing classification effectiveness through various metrics.</li>
<li>Producing forecasts for sentences that have not been encountered before.</li>
<li>Acknowledging constraints linked to conventional machine learning methods for sentiment analysis.</li>
</ul>

<h2 align="left">13. Tools and libraries</h2>
The practical was carried out utilizing Python along with these libraries:Pandas – loading, manipulating, and analyzing datasets.
<ul>
  <li>NumPy – operations with numbers.</li>
   <li> NLTK – support for natural language processing.</li>
   <li> Scikit-learn – extracting features, training models, and assessing performance.</li>
  <li>  Matplotlib – graphical representation.</li>
  <li>  Seaborn – visualization of statistical data.</li>
</ul>
