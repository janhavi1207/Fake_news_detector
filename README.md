# Fake_news_detector
<h1 align="center">Fake News Detector</h1>
<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>
<h2>Introduction</h2>
<p>Fake News Detector is a simple Python script for classifying news articles as either "FAKE" or "REAL" using the Passive Aggressive Classifier and TF-IDF (Term Frequency-Inverse Document Frequency) features. The script reads data from a CSV file named 'news.csv', performs data preprocessing, trains the classifier, makes predictions, and evaluates the model's accuracy.</p>
<h2>Dependencies</h2>
<p>To run the script, you need the following Python libraries installed in your environment:</p>
<ul>
  <li>NumPy</li>
  <li>pandas</li>
  <li>scikit-learn</li>
</ul>
<p>Install the required dependencies using the following command:</p>

```
pip install numpy pandas scikit-learn
```
<h2>Usage</h2>
<ol>
  <li>Prepare your data: Ensure you have a CSV file named 'news.csv' containing labeled news articles. The 'news.csv' file should have two columns - 'text' containing the content of the news articles, and 'label' with the corresponding labels ('FAKE' or 'REAL').</li>
  <li>Clone or download the script to your local machine.</li>
  <li>Update the path to the 'news.csv' file in the script. Modify the following line with the correct path:</li>
</ol>

```
df = pd.read_csv('path/to/your/news.csv')
```
<ol start="4">
  <li>Run the script using Python:</li>
</ol>

```
python fake_news_detector.py
```
<ol start="5">
  <li>The script will read the data, split it into training and testing sets, build the classifier, make predictions, and print the accuracy of the model along with a confusion matrix to evaluate its performance.</li>
</ol>
<h2>License</h2>
<p>This project is licensed under the <a href="https://www.apache.org/licenses/LICENSE-2.0">Apache License 2.0</a>.</p>

