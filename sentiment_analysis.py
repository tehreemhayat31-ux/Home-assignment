#!/usr/bin/env python3
"""
YouTube Comments Sentiment Analysis
Analyzes comments for positive, negative, and neutral sentiments with regression analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class YouTubeSentimentAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load CSV data and clean it"""
        print("Loading and cleaning data...")
        self.df = pd.read_csv(self.csv_file)
        
        # Clean text data
        self.df['text_clean'] = self.df['text'].astype(str).apply(self.clean_text)
        
        # Convert votes to numeric
        self.df['votes_numeric'] = pd.to_numeric(self.df['votes'].str.replace(r'[^\d]', '', regex=True), errors='coerce').fillna(0)
        
        print(f"Loaded {len(self.df)} comments")
        return self.df
    
    def clean_text(self, text):
        """Clean text for sentiment analysis"""
        if pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep emojis
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def analyze_sentiment(self):
        """Perform sentiment analysis on comments"""
        print("Performing sentiment analysis...")
        
        sentiments = []
        polarities = []
        subjectivities = []
        
        for text in self.df['text_clean']:
            if text.strip():
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Categorize sentiment
                if polarity > 0.1:
                    sentiment = 'Positive'
                elif polarity < -0.1:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                sentiments.append(sentiment)
                polarities.append(polarity)
                subjectivities.append(subjectivity)
            else:
                sentiments.append('Neutral')
                polarities.append(0)
                subjectivities.append(0)
        
        self.df['sentiment'] = sentiments
        self.df['polarity'] = polarities
        self.df['subjectivity'] = subjectivities
        
        return self.df
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        print("Generating statistics...")
        
        # Basic sentiment distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        sentiment_percentages = (sentiment_counts / len(self.df)) * 100
        
        # Average polarity and subjectivity by sentiment
        sentiment_stats = self.df.groupby('sentiment').agg({
            'polarity': ['mean', 'std'],
            'subjectivity': ['mean', 'std'],
            'votes_numeric': ['mean', 'sum', 'count']
        }).round(3)
        
        # Top comments by polarity (most positive/negative)
        most_positive = self.df.nlargest(10, 'polarity')[['author', 'text', 'polarity', 'votes_numeric']]
        most_negative = self.df.nsmallest(10, 'polarity')[['author', 'text', 'polarity', 'votes_numeric']]
        
        # Most liked comments
        most_liked = self.df.nlargest(10, 'votes_numeric')[['author', 'text', 'sentiment', 'votes_numeric']]
        
        self.results = {
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'sentiment_stats': sentiment_stats,
            'most_positive': most_positive,
            'most_negative': most_negative,
            'most_liked': most_liked
        }
        
        return self.results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('YouTube Comments Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#2E8B57', '#DC143C', '#FFD700']
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Sentiment Distribution', fontweight='bold')
        
        # 2. Polarity Distribution Histogram
        axes[0, 1].hist(self.df['polarity'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Polarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Polarity Distribution', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Subjectivity vs Polarity Scatter Plot
        scatter = axes[0, 2].scatter(self.df['subjectivity'], self.df['polarity'], 
                                   c=self.df['polarity'], cmap='RdYlGn', alpha=0.6)
        axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('Subjectivity')
        axes[0, 2].set_ylabel('Polarity')
        axes[0, 2].set_title('Subjectivity vs Polarity', fontweight='bold')
        plt.colorbar(scatter, ax=axes[0, 2])
        
        # 4. Sentiment vs Votes Box Plot
        sentiment_votes = self.df[self.df['votes_numeric'] > 0]  # Only comments with votes
        if len(sentiment_votes) > 0:
            sentiment_votes.boxplot(column='votes_numeric', by='sentiment', ax=axes[1, 0])
            axes[1, 0].set_title('Votes Distribution by Sentiment', fontweight='bold')
            axes[1, 0].set_xlabel('Sentiment')
            axes[1, 0].set_ylabel('Number of Votes')
        else:
            axes[1, 0].text(0.5, 0.5, 'No votes data available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Votes Distribution by Sentiment', fontweight='bold')
        
        # 5. Average Polarity by Sentiment
        avg_polarity = self.df.groupby('sentiment')['polarity'].mean()
        bars = axes[1, 1].bar(avg_polarity.index, avg_polarity.values, color=colors)
        axes[1, 1].set_title('Average Polarity by Sentiment', fontweight='bold')
        axes[1, 1].set_ylabel('Average Polarity')
        axes[1, 1].set_ylim(-1, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_polarity.values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Sentiment Trend Over Time (if time data available)
        if 'time_parsed' in self.df.columns:
            try:
                # Convert timestamp to datetime
                self.df['datetime'] = pd.to_datetime(self.df['time_parsed'], unit='s')
                time_sentiment = self.df.groupby([self.df['datetime'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                time_sentiment.plot(kind='line', ax=axes[1, 2], marker='o')
                axes[1, 2].set_title('Sentiment Trend Over Time', fontweight='bold')
                axes[1, 2].set_xlabel('Date')
                axes[1, 2].set_ylabel('Number of Comments')
                axes[1, 2].legend()
                axes[1, 2].tick_params(axis='x', rotation=45)
            except:
                axes[1, 2].text(0.5, 0.5, 'Time trend not available', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Sentiment Trend Over Time', fontweight='bold')
        else:
            axes[1, 2].text(0.5, 0.5, 'Time data not available', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Sentiment Trend Over Time', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def print_detailed_report(self):
        """Print detailed analysis report"""
        print("\n" + "="*80)
        print("📊 YOUTUBE COMMENTS SENTIMENT ANALYSIS REPORT")
        print("="*80)
        
        # Overall Statistics
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"Total Comments Analyzed: {len(self.df):,}")
        print(f"Average Polarity: {self.df['polarity'].mean():.3f}")
        print(f"Average Subjectivity: {self.df['subjectivity'].mean():.3f}")
        
        # Sentiment Distribution
        print(f"\n🎯 SENTIMENT DISTRIBUTION:")
        for sentiment, count in self.results['sentiment_counts'].items():
            percentage = self.results['sentiment_percentages'][sentiment]
            print(f"  {sentiment}: {count:,} comments ({percentage:.1f}%)")
        
        # Sentiment Statistics
        print(f"\n📊 SENTIMENT STATISTICS:")
        print(self.results['sentiment_stats'])
        
        # Most Positive Comments
        print(f"\n😊 TOP 5 MOST POSITIVE COMMENTS:")
        for idx, row in self.results['most_positive'].head().iterrows():
            print(f"  Author: {row['author']}")
            print(f"  Comment: {row['text'][:100]}...")
            print(f"  Polarity: {row['polarity']:.3f}, Votes: {row['votes_numeric']}")
            print()
        
        # Most Negative Comments
        print(f"\n😞 TOP 5 MOST NEGATIVE COMMENTS:")
        for idx, row in self.results['most_negative'].head().iterrows():
            print(f"  Author: {row['author']}")
            print(f"  Comment: {row['text'][:100]}...")
            print(f"  Polarity: {row['polarity']:.3f}, Votes: {row['votes_numeric']}")
            print()
        
        # Most Liked Comments
        print(f"\n👍 TOP 5 MOST LIKED COMMENTS:")
        for idx, row in self.results['most_liked'].head().iterrows():
            print(f"  Author: {row['author']}")
            print(f"  Comment: {row['text'][:100]}...")
            print(f"  Sentiment: {row['sentiment']}, Votes: {row['votes_numeric']}")
            print()
        
        print("="*80)
    
    def save_results(self):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        results_file = f"sentiment_analysis_results_{timestamp}.csv"
        self.df.to_csv(results_file, index=False, encoding='utf-8')
        print(f"📁 Detailed results saved to: {results_file}")
        
        # Save summary statistics
        summary_file = f"sentiment_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("YOUTUBE COMMENTS SENTIMENT ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Comments: {len(self.df):,}\n")
            f.write(f"Average Polarity: {self.df['polarity'].mean():.3f}\n")
            f.write(f"Average Subjectivity: {self.df['subjectivity'].mean():.3f}\n\n")
            
            f.write("SENTIMENT DISTRIBUTION:\n")
            for sentiment, count in self.results['sentiment_counts'].items():
                percentage = self.results['sentiment_percentages'][sentiment]
                f.write(f"{sentiment}: {count:,} ({percentage:.1f}%)\n")
        
        print(f"📄 Summary saved to: {summary_file}")
        
        return results_file, summary_file

def main():
    # Find the CSV file
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'youtube_comments' in f]
    
    if not csv_files:
        print("❌ No YouTube comments CSV file found!")
        return
    
    csv_file = sorted(csv_files)[-1]  # Use the most recent file
    print(f"📂 Analyzing file: {csv_file}")
    
    # Initialize analyzer
    analyzer = YouTubeSentimentAnalyzer(csv_file)
    
    # Perform analysis
    analyzer.load_data()
    analyzer.analyze_sentiment()
    analyzer.generate_statistics()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Print report
    analyzer.print_detailed_report()
    
    # Save results
    analyzer.save_results()
    
    print("\n✅ Sentiment analysis completed successfully!")
    print("📊 Check the generated files and visualizations for detailed results.")

if __name__ == "__main__":
    main() 