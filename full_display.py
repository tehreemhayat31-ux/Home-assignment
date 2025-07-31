#!/usr/bin/env python3
"""
Full Display of YouTube Comments Analysis
Shows ALL results comprehensively on the main screen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

def load_and_analyze(csv_file):
    """Load and analyze data"""
    print("🔄 Loading and analyzing data...")
    
    df = pd.read_csv(csv_file)
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    df['votes_numeric'] = pd.to_numeric(df['votes'].str.replace(r'[^\d]', '', regex=True), errors='coerce').fillna(0)
    df['text_length'] = df['text_clean'].str.len()
    
    # Sentiment analysis
    sentiments = []
    polarities = []
    subjectivities = []
    
    for text in df['text_clean']:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        sentiments.append(sentiment)
        polarities.append(polarity)
        subjectivities.append(subjectivity)
    
    df['sentiment'] = sentiments
    df['polarity'] = polarities
    df['subjectivity'] = subjectivities
    
    return df

def clean_text(text):
    """Clean text"""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]', ' ', text)
    return ' '.join(text.split()).strip()

def perform_linear_regression(df):
    """Perform linear regression analysis"""
    print("📈 Performing linear regression analysis...")
    
    # Prepare data for regression
    regression_data = df[df['votes_numeric'] > 0].copy()
    
    if len(regression_data) < 10:
        return None
    
    # Multiple regression: Predict votes based on polarity, subjectivity, and text length
    X = regression_data[['polarity', 'subjectivity', 'text_length']].copy()
    y = regression_data['votes_numeric']
    
    # Fit regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Correlation analysis
    correlations = {}
    for col in ['polarity', 'subjectivity', 'text_length']:
        corr, p_value = stats.pearsonr(regression_data[col], y)
        correlations[col] = {'correlation': corr, 'p_value': p_value}
    
    regression_results = {
        'model': model,
        'coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': model.intercept_,
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'correlations': correlations,
        'data_points': len(regression_data),
        'predictions': y_pred,
        'actual': y.values
    }
    
    return regression_results

def display_full_analysis(df, regression_results=None):
    """Display FULL analysis results prominently on terminal"""
    
    # Calculate all statistics
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percentages = (sentiment_counts / len(df)) * 100
    total_votes = df['votes_numeric'].sum()
    avg_votes = df['votes_numeric'].mean()
    comments_with_votes = (df['votes_numeric'] > 0).sum()
    engagement_rate = (comments_with_votes / len(df)) * 100
    
    # Clear screen and display prominently
    os.system('clear')
    
    print("\n" + "="*120)
    print("🎵 COMPLETE YOUTUBE COMMENTS ANALYSIS RESULTS 🎵")
    print("="*120)
    
    # 1. MAIN STATISTICS
    print(f"\n📊 MAIN STATISTICS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"║ Total Comments Analyzed: {len(df):,}                                                                                    ║")
    print(f"║ Average Polarity: {df['polarity'].mean():.3f}                                                                                    ║")
    print(f"║ Average Subjectivity: {df['subjectivity'].mean():.3f}                                                                                ║")
    print(f"║ Total Engagement (Votes): {total_votes:,}                                                                                ║")
    print(f"║ Average Text Length: {df['text_length'].mean():.1f} characters                                                                        ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # 2. SENTIMENT DISTRIBUTION
    print(f"\n🎯 SENTIMENT DISTRIBUTION:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = sentiment_percentages.get(sentiment, 0)
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        if sentiment == 'Positive':
            emoji = '😊'
            color_indicator = '🟢'
        elif sentiment == 'Negative':
            emoji = '😞'
            color_indicator = '🔴'
        else:
            emoji = '😐'
            color_indicator = '🟡'
        
        print(f"║ {emoji} {sentiment:8} Comments: {count:5,} ({percentage:5.1f}%) {color_indicator} {bar} ║")
    
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # 3. ENGAGEMENT STATISTICS
    print(f"\n📈 ENGAGEMENT STATISTICS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"║ Total Votes: {total_votes:>8,}                                                                                    ║")
    print(f"║ Average Votes per Comment: {avg_votes:>6.2f}                                                                                ║")
    print(f"║ Comments with Votes: {comments_with_votes:>6,}                                                                                ║")
    print(f"║ Engagement Rate: {engagement_rate:>6.1f}%                                                                                    ║")
    print(f"║ Max Votes on Single Comment: {df['votes_numeric'].max():>6,}                                                                                ║")
    print(f"║ Min Votes on Single Comment: {df['votes_numeric'].min():>6,}                                                                                ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # 4. SENTIMENT BY VOTES DETAILED
    print(f"\n📊 SENTIMENT BY VOTES (DETAILED):")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    sentiment_vote_stats = df.groupby('sentiment')['votes_numeric'].agg(['mean', 'std', 'sum', 'count', 'min', 'max']).round(2)
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in sentiment_vote_stats.index:
            stats = sentiment_vote_stats.loc[sentiment]
            print(f"║ {sentiment:8} | Avg: {stats['mean']:>6.2f} | Std: {stats['std']:>6.2f} | Total: {stats['sum']:>6,} | Count: {stats['count']:>4,} | Min: {stats['min']:>3} | Max: {stats['max']:>3} ║")
    
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # 5. TEXT STATISTICS
    print(f"\n📝 TEXT STATISTICS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"║ Average Text Length: {df['text_length'].mean():>6.1f} characters                                                                    ║")
    print(f"║ Maximum Text Length: {df['text_length'].max():>6,} characters                                                                    ║")
    print(f"║ Minimum Text Length: {df['text_length'].min():>6,} characters                                                                    ║")
    print(f"║ Total Characters: {df['text_length'].sum():>8,}                                                                                ║")
    print(f"║ Text Length Std Dev: {df['text_length'].std():>6.1f} characters                                                                    ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # 6. POLARITY AND SUBJECTIVITY STATISTICS
    print(f"\n📊 POLARITY AND SUBJECTIVITY STATISTICS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"║ Polarity - Mean: {df['polarity'].mean():>6.3f} | Std: {df['polarity'].std():>6.3f} | Min: {df['polarity'].min():>6.3f} | Max: {df['polarity'].max():>6.3f} ║")
    print(f"║ Subjectivity - Mean: {df['subjectivity'].mean():>6.3f} | Std: {df['subjectivity'].std():>6.3f} | Min: {df['subjectivity'].min():>6.3f} | Max: {df['subjectivity'].max():>6.3f} ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # 7. LINEAR REGRESSION ANALYSIS
    if regression_results:
        print(f"\n📈 LINEAR REGRESSION ANALYSIS:")
        print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        reg = regression_results
        print(f"║ Data Points Used: {reg['data_points']:,}                                                                                ║")
        print(f"║ R² Score: {reg['r2_score']:.3f}                                                                                    ║")
        print(f"║ Mean Squared Error: {reg['mse']:.3f}                                                                                ║")
        print(f"║ Root Mean Squared Error: {reg['rmse']:.3f}                                                                                ║")
        print(f"║ Intercept: {reg['intercept']:.3f}                                                                                    ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
        
        print(f"\n📊 REGRESSION COEFFICIENTS:")
        print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        for feature, coef in reg['coefficients'].items():
            print(f"║ {feature:12}: {coef:>8.3f}                                                                                    ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
        
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        for feature, corr_data in reg['correlations'].items():
            significance = "***" if corr_data['p_value'] < 0.001 else "**" if corr_data['p_value'] < 0.01 else "*" if corr_data['p_value'] < 0.05 else ""
            print(f"║ {feature:12}: r = {corr_data['correlation']:>6.3f}, p = {corr_data['p_value']:>6.3f} {significance:<3} ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
        
        print(f"\n📋 REGRESSION EQUATION:")
        print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        equation = f"Votes = {reg['intercept']:.3f}"
        for feature, coef in reg['coefficients'].items():
            if coef >= 0:
                equation += f" + {coef:.3f}×{feature}"
            else:
                equation += f" - {abs(coef):.3f}×{feature}"
        print(f"║ {equation:<88} ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # 8. TOP COMMENTS BY DIFFERENT METRICS
    print(f"\n😊 TOP 5 MOST POSITIVE COMMENTS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    top_positive = df.nlargest(5, 'polarity')[['author', 'text', 'polarity', 'votes_numeric']]
    for idx, row in top_positive.iterrows():
        print(f"║ Author: {row['author'][:30]:<30} | Polarity: {row['polarity']:>6.3f} | Votes: {row['votes_numeric']:>3} ║")
        print(f"║ Comment: {row['text'][:70]:<70}... ║")
        print(f"║ {'':<88} ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n😞 TOP 5 MOST NEGATIVE COMMENTS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    top_negative = df.nsmallest(5, 'polarity')[['author', 'text', 'polarity', 'votes_numeric']]
    for idx, row in top_negative.iterrows():
        print(f"║ Author: {row['author'][:30]:<30} | Polarity: {row['polarity']:>6.3f} | Votes: {row['votes_numeric']:>3} ║")
        print(f"║ Comment: {row['text'][:70]:<70}... ║")
        print(f"║ {'':<88} ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n👍 TOP 5 MOST LIKED COMMENTS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    top_liked = df.nlargest(5, 'votes_numeric')[['author', 'text', 'sentiment', 'votes_numeric']]
    for idx, row in top_liked.iterrows():
        print(f"║ Author: {row['author'][:30]:<30} | Sentiment: {row['sentiment']:<8} | Votes: {row['votes_numeric']:>3} ║")
        print(f"║ Comment: {row['text'][:70]:<70}... ║")
        print(f"║ {'':<88} ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n📝 TOP 5 LONGEST COMMENTS:")
    print(f"╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    top_longest = df.nlargest(5, 'text_length')[['author', 'text', 'text_length', 'sentiment', 'votes_numeric']]
    for idx, row in top_longest.iterrows():
        print(f"║ Author: {row['author'][:30]:<30} | Length: {row['text_length']:>4} | Sentiment: {row['sentiment']:<8} | Votes: {row['votes_numeric']:>3} ║")
        print(f"║ Comment: {row['text'][:70]:<70}... ║")
        print(f"║ {'':<88} ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n" + "="*120)
    print("🎉 COMPLETE ANALYSIS FINISHED! 🎉")
    print("="*120)

def create_and_save_visualizations(df, regression_results=None):
    """Create and save visualizations"""
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Complete YouTube Comments Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Sentiment Distribution
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#2E8B57', '#DC143C', '#FFD700']
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                  colors=colors, startangle=90)
    axes[0, 0].set_title('Sentiment Distribution', fontweight='bold')
    
    # 2. Polarity Distribution
    axes[0, 1].hist(df['polarity'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('Polarity Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Polarity Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Subjectivity vs Polarity
    scatter = axes[0, 2].scatter(df['subjectivity'], df['polarity'], 
                               c=df['polarity'], cmap='RdYlGn', alpha=0.6)
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 2].set_xlabel('Subjectivity')
    axes[0, 2].set_ylabel('Polarity')
    axes[0, 2].set_title('Subjectivity vs Polarity', fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 2])
    
    # 4. Votes Distribution
    votes_data = df[df['votes_numeric'] > 0]['votes_numeric']
    axes[1, 0].hist(votes_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Votes')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Votes Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Sentiment vs Votes
    sentiment_votes = df[df['votes_numeric'] > 0]
    sentiment_votes.boxplot(column='votes_numeric', by='sentiment', ax=axes[1, 1])
    axes[1, 1].set_title('Votes by Sentiment', fontweight='bold')
    axes[1, 1].set_xlabel('Sentiment')
    axes[1, 1].set_ylabel('Number of Votes')
    
    # 6. Text Length vs Votes
    text_votes = df[df['votes_numeric'] > 0]
    axes[1, 2].scatter(text_votes['text_length'], text_votes['votes_numeric'], alpha=0.6)
    axes[1, 2].set_xlabel('Text Length')
    axes[1, 2].set_ylabel('Number of Votes')
    axes[1, 2].set_title('Text Length vs Votes', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Regression Analysis - Polarity vs Votes
    if regression_results:
        reg_data = df[df['votes_numeric'] > 0]
        axes[2, 0].scatter(reg_data['polarity'], reg_data['votes_numeric'], alpha=0.6, color='blue')
        
        # Add regression line
        z = np.polyfit(reg_data['polarity'], reg_data['votes_numeric'], 1)
        p = np.poly1d(z)
        axes[2, 0].plot(reg_data['polarity'], p(reg_data['polarity']), "r--", alpha=0.8)
        
        axes[2, 0].set_xlabel('Polarity')
        axes[2, 0].set_ylabel('Number of Votes')
        axes[2, 0].set_title(f'Polarity vs Votes (R² = {regression_results["r2_score"]:.3f})', fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Correlation Heatmap
    numeric_cols = ['polarity', 'subjectivity', 'votes_numeric', 'text_length']
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[2, 1])
    axes[2, 1].set_title('Correlation Matrix', fontweight='bold')
    
    # 9. Average Metrics by Sentiment
    avg_metrics = df.groupby('sentiment').agg({
        'polarity': 'mean',
        'subjectivity': 'mean',
        'votes_numeric': 'mean'
    })
    
    x = np.arange(len(avg_metrics.index))
    width = 0.25
    
    axes[2, 2].bar(x - width, avg_metrics['polarity'], width, label='Polarity', alpha=0.8)
    axes[2, 2].bar(x, avg_metrics['subjectivity'], width, label='Subjectivity', alpha=0.8)
    axes[2, 2].bar(x + width, avg_metrics['votes_numeric'], width, label='Avg Votes', alpha=0.8)
    
    axes[2, 2].set_xlabel('Sentiment')
    axes[2, 2].set_ylabel('Average Score')
    axes[2, 2].set_title('Average Metrics by Sentiment', fontweight='bold')
    axes[2, 2].set_xticks(x)
    axes[2, 2].set_xticklabels(avg_metrics.index)
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_display_analysis_results.png', dpi=300, bbox_inches='tight')
    print("📁 Visualizations saved as: full_display_analysis_results.png")
    
    return fig

def main():
    # Find CSV file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'youtube_comments' in f]
    csv_file = sorted(csv_files)[-1]
    
    print(f"📂 Analyzing file: {csv_file}")
    
    # Load and analyze
    df = load_and_analyze(csv_file)
    
    # Perform linear regression
    regression_results = perform_linear_regression(df)
    
    # Display FULL results prominently
    display_full_analysis(df, regression_results)
    
    # Create and save visualizations
    create_and_save_visualizations(df, regression_results)
    
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"full_display_analysis_results_{timestamp}.csv"
    df.to_csv(results_file, index=False, encoding='utf-8')
    
    print(f"\n📁 Complete analysis results saved to: {results_file}")
    print("✅ Full display analysis completed successfully!")

if __name__ == "__main__":
    main() 