#!/usr/bin/env python3
"""
MSBuild Backlog Analysis Tool
Analyzes issues for duplicates, categorization, and staleness
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.void,)):
            return None
        return json.JSONEncoder.default(self, obj)

class BacklogAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.issues = []
        self.df = None
        self.duplicate_groups = []
        self.topic_clusters = []
        self.outdated_issues = pd.DataFrame()  # Initialize as empty DataFrame instead of list
        
    def load_issues(self):
        """Load all issues from JSON files"""
        backlog_dir = self.data_dir / 'backlog'
        issue_files = list(backlog_dir.glob('issue-*.json'))
        
        print(f"Loading {len(issue_files)} issues...")
        
        for issue_file in issue_files:
            with open(issue_file, 'r', encoding='utf-8') as f:
                issue_data = json.load(f)
                
                # Extract key fields
                issue = {
                    'number': issue_data.get('number'),
                    'title': issue_data.get('title', ''),
                    'body': issue_data.get('body', ''),
                    'state': issue_data.get('state', ''),
                    'created_at': issue_data.get('created_at', ''),
                    'updated_at': issue_data.get('updated_at', ''),
                    'closed_at': issue_data.get('closed_at', ''),
                    'author': issue_data.get('author', {}).get('login', '') if issue_data.get('author') else '',
                    'labels': issue_data.get('labels', []),
                    'milestone': issue_data.get('milestone', {}).get('title', '') if issue_data.get('milestone') else '',
                    'assignees': [a.get('login', '') for a in issue_data.get('assignees', {}).get('nodes', [])],
                    'comments_count': len(issue_data.get('comments', {}).get('nodes', [])),
                    'comments': issue_data.get('comments', {}).get('nodes', [])
                }
                
                # Calculate derived fields
                if issue['created_at']:
                    try:
                        issue['created_date'] = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                    except:
                        issue['created_date'] = None
                else:
                    issue['created_date'] = None
                    
                if issue['updated_at']:
                    try:
                        issue['updated_date'] = datetime.fromisoformat(issue['updated_at'].replace('Z', '+00:00'))
                    except:
                        issue['updated_date'] = None
                else:
                    issue['updated_date'] = None
                    
                if issue['closed_at']:
                    try:
                        issue['closed_date'] = datetime.fromisoformat(issue['closed_at'].replace('Z', '+00:00'))
                    except:
                        issue['closed_date'] = None
                else:
                    issue['closed_date'] = None
                
                self.issues.append(issue)
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.issues)
        print(f"Loaded {len(self.df)} issues")
        
    def analyze_duplicates(self, similarity_threshold=0.7):
        """Find potential duplicate issues using text similarity"""
        print("\n=== DUPLICATE ANALYSIS ===")
        
        # Combine title and body for similarity analysis
        self.df['text'] = self.df['title'] + ' ' + self.df['body'].fillna('')
        
        # Filter only open issues
        open_issues = self.df[self.df['state'] == 'OPEN'].copy()
        
        if len(open_issues) == 0:
            print("No open issues found")
            return []
        
        print(f"Analyzing {len(open_issues)} open issues for duplicates...")
        
        # Create TF-IDF vectors with enhanced parameters
        vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(open_issues['text'])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicates with enhanced grouping
        duplicates = []
        processed = set()
        
        for i in range(len(open_issues)):
            if i in processed:
                continue
                
            # Find all similar issues
            similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
            similar_indices = [idx for idx in similar_indices if idx != i and idx not in processed]
            
            if similar_indices:
                group = [i] + list(similar_indices)
                processed.update(group)
                
                # Calculate group statistics
                group_similarities = []
                for idx1 in group:
                    for idx2 in group:
                        if idx1 < idx2:
                            group_similarities.append(similarity_matrix[idx1][idx2])
                
                duplicate_group = {
                    'issues': [
                        {
                            'number': int(open_issues.iloc[idx]['number']),
                            'title': str(open_issues.iloc[idx]['title']),
                            'created_at': str(open_issues.iloc[idx]['created_at']),
                            'labels': open_issues.iloc[idx]['labels'],
                            'author': str(open_issues.iloc[idx]['author']),
                            'similarity': float(similarity_matrix[i][idx]) if idx != i else 1.0
                        }
                        for idx in group
                    ],
                    'max_similarity': float(max(similarity_matrix[i][idx] for idx in similar_indices)),
                    'avg_similarity': float(np.mean(group_similarities)) if group_similarities else 1.0,
                    'group_size': len(group),
                    'common_keywords': self._extract_common_keywords(open_issues.iloc[group], vectorizer)
                }
                duplicates.append(duplicate_group)
        
        # Sort by average similarity and group size
        duplicates.sort(key=lambda x: (x['avg_similarity'], x['group_size']), reverse=True)
        
        self.duplicate_groups = duplicates
        
        print(f"Found {len(duplicates)} potential duplicate groups")
        
        # Enhanced duplicate reporting
        for i, dup_group in enumerate(duplicates[:10]):
            print(f"\nDuplicate Group {i+1} (avg similarity: {dup_group['avg_similarity']:.2f}, size: {dup_group['group_size']}):")
            print(f"  Common keywords: {', '.join(dup_group['common_keywords'][:5])}")
            for issue in dup_group['issues']:
                print(f"  #{issue['number']}: {issue['title'][:80]}...")
                print(f"    Author: {issue['author']}, Similarity: {issue['similarity']:.2f}")
        
        return duplicates
    
    def _extract_common_keywords(self, issues_df, vectorizer):
        """Extract common keywords from a group of issues"""
        combined_text = ' '.join(issues_df['text'].values)
        tfidf_vector = vectorizer.transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords
        scores = tfidf_vector.toarray()[0]
        top_indices = scores.argsort()[-10:][::-1]
        
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    
    def analyze_overlapping_topics(self, n_topics=10):
        """Analyze overlapping topics using topic modeling"""
        print("\n=== OVERLAPPING TOPICS ANALYSIS ===")
        
        # Prepare text data
        self.df['text'] = self.df['title'] + ' ' + self.df['body'].fillna('')
        open_issues = self.df[self.df['state'] == 'OPEN'].copy()
        
        if len(open_issues) < n_topics:
            print(f"Not enough open issues for {n_topics} topics")
            return {}
        
        print(f"Analyzing {len(open_issues)} open issues for topic overlap...")
        
        # Create document-term matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        doc_term_matrix = vectorizer.fit_transform(open_issues['text'])
        
        # Apply LDA for topic modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        doc_topic_matrix = lda.fit_transform(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-20:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[f"Topic_{topic_idx}"] = {
                'words': top_words[:10],
                'weight': float(topic[top_indices].sum())
            }
        
        # Assign issues to topics
        open_issues['primary_topic'] = doc_topic_matrix.argmax(axis=1)
        open_issues['topic_score'] = doc_topic_matrix.max(axis=1)
        
        # Find overlapping issues (high scores in multiple topics)
        overlap_threshold = 0.3
        overlapping_issues = []
        
        for idx, scores in enumerate(doc_topic_matrix):
            high_score_topics = np.where(scores > overlap_threshold)[0]
            if len(high_score_topics) > 1:
                issue_idx = open_issues.index[idx]
                overlapping_issues.append({
                    'number': int(self.df.loc[issue_idx, 'number']),
                    'title': str(self.df.loc[issue_idx, 'title']),
                    'topics': [int(t) for t in high_score_topics],
                    'topic_scores': {int(t): float(scores[t]) for t in high_score_topics}
                })
        
        # Print topic analysis
        print(f"\nIdentified {n_topics} topics:")
        for topic_name, topic_info in topics.items():
            print(f"\n{topic_name}:")
            print(f"  Keywords: {', '.join(topic_info['words'][:5])}")
            topic_issues = open_issues[open_issues['primary_topic'] == int(topic_name.split('_')[1])]
            print(f"  Issues: {len(topic_issues)}")
        
        print(f"\nFound {len(overlapping_issues)} issues with overlapping topics")
        
        # Show examples of overlapping issues
        for issue in overlapping_issues[:5]:
            print(f"\n#{issue['number']}: {issue['title'][:80]}...")
            print(f"  Topics: {issue['topics']} with scores: {issue['topic_scores']}")
        
        self.topic_clusters = {
            'topics': topics,
            'overlapping_issues': overlapping_issues,
            'topic_distribution': open_issues['primary_topic'].value_counts().to_dict()
        }
        
        return self.topic_clusters
    
    def analyze_outdated_issues(self, stale_days=180):
        """Enhanced analysis of outdated and stale issues"""
        print("\n=== OUTDATED ISSUES ANALYSIS ===")
        
        # Filter out issues with missing dates
        df_with_dates = self.df[self.df['updated_date'].notna()].copy()
        
        if len(df_with_dates) == 0:
            print("No issues with valid update dates found")
            return pd.DataFrame()
        
        # Get current time with proper timezone handling
        sample_date = df_with_dates['updated_date'].iloc[0]
        if hasattr(sample_date, 'tzinfo') and sample_date.tzinfo is not None:
            now = datetime.now(tz=sample_date.tzinfo)
        else:
            now = datetime.now()
        
        # Find stale open issues
        open_issues = df_with_dates[df_with_dates['state'] == 'OPEN'].copy()
        
        if len(open_issues) == 0:
            print("No open issues found")
            return pd.DataFrame()
        
        # Calculate various staleness metrics
        open_issues['days_since_update'] = open_issues['updated_date'].apply(
            lambda x: (now - x).days if pd.notna(x) else None
        )
        open_issues['days_since_creation'] = open_issues['created_date'].apply(
            lambda x: (now - x).days if pd.notna(x) else None
        )
        
        # Identify different types of outdated issues
        outdated_criteria = {
            'stale': open_issues['days_since_update'] >= stale_days,
            'very_old': open_issues['days_since_creation'] >= 730,  # 2+ years old
            'abandoned': (open_issues['days_since_update'] >= 365) & (open_issues['comments_count'] == 0),
            'no_assignee': (open_issues['days_since_update'] >= 90) & (open_issues['assignees'].apply(len) == 0),
            'no_milestone': (open_issues['days_since_update'] >= 90) & (open_issues['milestone'] == '')
        }
        
        # Create outdated issues dataframe
        outdated_issues = pd.DataFrame()
        outdated_reasons = []
        
        for reason, mask in outdated_criteria.items():
            matching_issues = open_issues[mask].copy()
            matching_issues['outdated_reason'] = reason
            outdated_reasons.append(matching_issues)
        
        if outdated_reasons:
            outdated_issues = pd.concat(outdated_reasons).drop_duplicates(subset=['number'])
        
        # Add outdated score
        outdated_issues['outdated_score'] = 0
        for reason, mask in outdated_criteria.items():
            outdated_issues.loc[mask, 'outdated_score'] += 1
        
        # Sort by outdated score and days since update
        outdated_issues = outdated_issues.sort_values(
            ['outdated_score', 'days_since_update'], 
            ascending=[False, False]
        )
        
        self.outdated_issues = outdated_issues
        
        # Print analysis
        print(f"Found {len(outdated_issues)} outdated issues")
        
        print("\nOutdated issues by criteria:")
        for reason, mask in outdated_criteria.items():
            count = mask.sum()
            print(f"  {reason}: {count} issues")
        
        # Most outdated issues
        print("\nMost outdated issues (multiple criteria):")
        high_score = outdated_issues[outdated_issues['outdated_score'] >= 2]
        for _, issue in high_score.head(10).iterrows():
            print(f"  #{issue['number']}: {issue['title'][:60]}...")
            print(f"    Score: {issue['outdated_score']}, "
                  f"Last updated: {issue['days_since_update']:.0f} days ago, "
                  f"Age: {issue['days_since_creation']:.0f} days")
        
        return outdated_issues
    
    def analyze_cross_references(self):
        """Analyze cross-references between duplicates, overlapping topics, and outdated issues"""
        print("\n=== CROSS-REFERENCE ANALYSIS ===")
        
        if not self.duplicate_groups or self.outdated_issues.empty:
            print("Need to run duplicate and outdated analysis first")
            return
        
        # Get issue numbers from each category
        duplicate_numbers = set()
        for group in self.duplicate_groups:
            for issue in group['issues']:
                duplicate_numbers.add(issue['number'])
        
        outdated_numbers = set(self.outdated_issues['number'].values)
        
        # Find intersections
        duplicate_and_outdated = duplicate_numbers & outdated_numbers
        
        print(f"\nIssues that are both duplicate and outdated: {len(duplicate_and_outdated)}")
        
        if duplicate_and_outdated:
            print("\nExamples of duplicate AND outdated issues:")
            for num in list(duplicate_and_outdated)[:5]:
                issue = self.df[self.df['number'] == num].iloc[0]
                print(f"  #{num}: {issue['title'][:80]}...")
        
        # Analyze topic overlap in duplicates
        if hasattr(self, 'topic_clusters') and self.topic_clusters:
            overlapping_numbers = {issue['number'] for issue in self.topic_clusters['overlapping_issues']}
            
            dup_and_overlap = duplicate_numbers & overlapping_numbers
            print(f"\nIssues with duplicate content AND overlapping topics: {len(dup_and_overlap)}")
            
            all_three = duplicate_numbers & outdated_numbers & overlapping_numbers
            print(f"\nIssues in ALL THREE categories: {len(all_three)}")
            
            if all_three:
                print("\nHighest priority issues (in all categories):")
                for num in list(all_three)[:5]:
                    issue = self.df[self.df['number'] == num].iloc[0]
                    print(f"  #{num}: {issue['title'][:80]}...")
    
    def generate_report(self, output_file='backlog_analysis_report.md'):
        """Generate a comprehensive markdown report"""
        print(f"\nGenerating report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# MSBuild Backlog Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Total issues analyzed: {len(self.df)}\n")
            f.write(f"- Open issues: {len(self.df[self.df['state'] == 'OPEN'])}\n")
            f.write(f"- Closed issues: {len(self.df[self.df['state'] == 'CLOSED'])}\n\n")
            
            # Add more sections as needed
            
        print(f"Report generated: {output_file}")
    
    def visualize_insights(self):
        """Create visualizations of the backlog analysis"""
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        # Create output directory
        viz_dir = Path('backlog_visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Issues over time
        df_with_dates = self.df[self.df['created_date'].notna()].copy()
        if len(df_with_dates) > 0:
            plt.figure(figsize=(12, 6))
            df_with_dates.groupby([df_with_dates['created_date'].dt.to_period('M'), 'state']).size().unstack(fill_value=0).plot(kind='bar', stacked=True)
            plt.title('Issues Created Over Time')
            plt.xlabel('Month')
            plt.ylabel('Number of Issues')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'issues_over_time.png')
            plt.close()
        
        # 2. Category distribution
        category_data = []
        for cats in self.df['categories']:
            for cat in cats:
                category_data.append(cat)
        
        if category_data:
            plt.figure(figsize=(10, 6))
            pd.Series(category_data).value_counts().plot(kind='bar')
            plt.title('Issue Categories Distribution')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'category_distribution.png')
            plt.close()
        
        print(f"Visualizations saved to {viz_dir}/")

    def categorize_issues(self):
        """Categorize issues based on labels, keywords, and content"""
        print("\n=== ISSUE CATEGORIZATION ===")
        
        # Define categories based on common patterns
        categories = {
            'bug': {
                'keywords': ['bug', 'error', 'crash', 'fail', 'broken', 'exception', 'incorrect'],
                'labels': ['bug', 'defect', 'issue']
            },
            'performance': {
                'keywords': ['slow', 'performance', 'speed', 'optimize', 'faster', 'efficient'],
                'labels': ['performance', 'optimization']
            },
            'feature': {
                'keywords': ['feature', 'enhancement', 'request', 'add', 'new', 'implement'],
                'labels': ['enhancement', 'feature-request', 'feature']
            },
            'documentation': {
                'keywords': ['document', 'docs', 'readme', 'guide', 'tutorial', 'example'],
                'labels': ['documentation', 'docs']
            },
            'build': {
                'keywords': ['build', 'compile', 'msbuild', 'project', 'target', 'task'],
                'labels': ['build', 'build-issue']
            },
            'compatibility': {
                'keywords': ['compatible', 'version', 'upgrade', 'migrate', 'breaking'],
                'labels': ['compatibility', 'breaking-change']
            },
            'test': {
                'keywords': ['test', 'unit test', 'testing', 'coverage', 'assertion'],
                'labels': ['test', 'testing']
            }
        }
        
        # Categorize each issue
        self.df['categories'] = self.df.apply(lambda row: self._categorize_issue(row, categories), axis=1)
        
        # Count categories
        category_counts = Counter()
        for cats in self.df['categories']:
            category_counts.update(cats)
        
        print("\nCategory Distribution:")
        for category, count in category_counts.most_common():
            percentage = (count / len(self.df)) * 100
            print(f"  {category}: {count} issues ({percentage:.1f}%)")
        
        # Uncategorized issues
        uncategorized = self.df[self.df['categories'].apply(len) == 0]
        print(f"\nUncategorized issues: {len(uncategorized)} ({len(uncategorized)/len(self.df)*100:.1f}%)")
        
        # Label analysis
        print("\nMost common labels:")
        all_labels = []
        for labels in self.df['labels']:
            all_labels.extend(labels)
        
        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common(15):
            print(f"  {label}: {count}")
        
        return category_counts
    
    def _categorize_issue(self, row, categories):
        """Categorize a single issue"""
        issue_categories = []
        
        # Check labels
        issue_labels = [label.lower() for label in row['labels']]
        
        # Check title and body
        text = (row['title'] + ' ' + str(row['body'])).lower()
        
        for category, criteria in categories.items():
            # Check labels
            if any(label in issue_labels for label in criteria['labels']):
                issue_categories.append(category)
                continue
            
            # Check keywords
            if any(keyword in text for keyword in criteria['keywords']):
                issue_categories.append(category)
        
        return issue_categories
    
    def analyze_issue_lifecycle(self):
        """Analyze issue lifecycle metrics"""
        print("\n=== ISSUE LIFECYCLE ANALYSIS ===")
        
        # Calculate time to close for closed issues
        closed_issues = self.df[
            (self.df['state'] == 'CLOSED') &
            (self.df['closed_date'].notna()) &
            (self.df['created_date'].notna())
        ].copy()

        if len(closed_issues) > 0:
            closed_issues['days_to_close'] = closed_issues.apply(
                lambda row: (row['closed_date'] - row['created_date']).days
                if pd.notna(row['closed_date']) and pd.notna(row['created_date'])
                else None, axis=1
            )
            
            # Filter out invalid values
            valid_days = closed_issues['days_to_close'].dropna()

            if len(valid_days) > 0:
                print(f"\nClosed Issues Statistics:")
                print(f"  Total closed: {len(closed_issues)}")
                print(f"  Average time to close: {valid_days.mean():.1f} days")
                print(f"  Median time to close: {valid_days.median():.1f} days")
                print(f"  Max time to close: {valid_days.max():.0f} days")
        
        # Open vs Closed ratio
        open_count = len(self.df[self.df['state'] == 'OPEN'])
        closed_count = len(self.df[self.df['state'] == 'CLOSED'])
        
        print(f"\nIssue State Distribution:")
        print(f"  Open: {open_count} ({open_count/len(self.df)*100:.1f}%)")
        print(f"  Closed: {closed_count} ({closed_count/len(self.df)*100:.1f}%)")
        
        # Issues by year
        df_with_dates = self.df[self.df['created_date'].notna()].copy()
        if len(df_with_dates) > 0:
            df_with_dates['created_year'] = df_with_dates['created_date'].dt.year
            yearly_counts = df_with_dates.groupby(['created_year', 'state']).size().unstack(fill_value=0)

            print("\nIssues by Year:")
            print(yearly_counts)
    
    def generate_enhanced_report(self, output_file='enhanced_backlog_analysis.md'):
        """Generate an enhanced markdown report with detailed analysis"""
        print(f"\nGenerating enhanced report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# MSBuild Enhanced Backlog Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- Total issues analyzed: {len(self.df)}\n")
            f.write(f"- Open issues: {len(self.df[self.df['state'] == 'OPEN'])}\n")
            f.write(f"- Duplicate groups found: {len(self.duplicate_groups)}\n")
            if not self.outdated_issues.empty:
                f.write(f"- Outdated issues: {len(self.outdated_issues)}\n")
            f.write("\n")
            
            # Duplicate Analysis
            f.write("## Duplicate Issues Analysis\n\n")
            if self.duplicate_groups:
                f.write(f"Found {len(self.duplicate_groups)} groups of potential duplicates.\n\n")
                
                for i, group in enumerate(self.duplicate_groups[:10]):
                    f.write(f"### Duplicate Group {i+1}\n")
                    f.write(f"- Average similarity: {group['avg_similarity']:.2f}\n")
                    f.write(f"- Group size: {group['group_size']}\n")
                    f.write(f"- Common keywords: {', '.join(group['common_keywords'][:5])}\n\n")
                    
                    f.write("Issues in this group:\n")
                    for issue in group['issues']:
                        f.write(f"- [#{issue['number']}]"
                               f"(https://github.com/dotnet/msbuild/issues/{issue['number']}): "
                               f"{issue['title']}\n")
                    f.write("\n")
            
            # Topic Overlap Analysis
            if hasattr(self, 'topic_clusters') and self.topic_clusters:
                f.write("## Topic Overlap Analysis\n\n")
                f.write(f"Identified {len(self.topic_clusters['topics'])} main topics.\n\n")
                
                for topic_name, topic_info in self.topic_clusters['topics'].items():
                    f.write(f"### {topic_name}\n")
                    f.write(f"- Keywords: {', '.join(topic_info['words'][:10])}\n")
                    f.write(f"- Number of issues: {self.topic_clusters['topic_distribution'].get(int(topic_name.split('_')[1]), 0)}\n\n")
                
                f.write(f"\n### Overlapping Issues\n")
                f.write(f"Found {len(self.topic_clusters['overlapping_issues'])} issues spanning multiple topics.\n\n")
            
            # Outdated Issues
            if not self.outdated_issues.empty:
                f.write("## Outdated Issues Analysis\n\n")
                
                # Group by outdated score
                score_groups = self.outdated_issues.groupby('outdated_score')
                for score, group in score_groups:
                    f.write(f"### Issues with Outdated Score: {int(score)}\n")
                    f.write(f"Total: {len(group)} issues\n\n")
                    
                    for _, issue in group.head(5).iterrows():
                        f.write(f"- [#{int(issue['number'])}]"
                               f"(https://github.com/dotnet/msbuild/issues/{int(issue['number'])}): "
                               f"{issue['title'][:80]}...\n")
                        f.write(f"  - Last updated: {issue['days_since_update']:.0f} days ago\n")
                        f.write(f"  - Created: {issue['days_since_creation']:.0f} days ago\n")
                    f.write("\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("1. **Duplicate Issues**: Review and consolidate duplicate groups to reduce noise\n")
            f.write("2. **Outdated Issues**: Close or update issues that haven't seen activity in over a year\n")
            f.write("3. **Topic Overlap**: Consider reorganizing issues with overlapping topics for better categorization\n")
            
        print(f"Enhanced report generated: {output_file}")

def main():
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / 'data' / 'raw-data' / 'msbuild'
    
    # Initialize analyzer
    analyzer = BacklogAnalyzer(data_dir)
    
    # Load issues
    analyzer.load_issues()
    
    if len(analyzer.df) == 0:
        print("No issues found to analyze!")
        return
    
    print(f"\nTotal issues loaded: {len(analyzer.df)}")

    # Run analyses
    try:
        duplicates = analyzer.analyze_duplicates(similarity_threshold=0.6)  # Lower threshold for more results
    except Exception as e:
        print(f"Error in duplicate analysis: {e}")
        duplicates = []
    
    try:
        topic_analysis = analyzer.analyze_overlapping_topics(n_topics=15)
    except Exception as e:
        print(f"Error in topic analysis: {e}")
        topic_analysis = {}
    
    try:
        categories = analyzer.categorize_issues()
    except Exception as e:
        print(f"Error in categorization: {e}")
        categories = {}

    try:
        outdated_issues = analyzer.analyze_outdated_issues(stale_days=180)
    except Exception as e:
        print(f"Error in outdated analysis: {e}")
        outdated_issues = pd.DataFrame()
    
    try:
        analyzer.analyze_cross_references()
    except Exception as e:
        print(f"Error in cross-reference analysis: {e}")

    try:
        analyzer.analyze_issue_lifecycle()
    except Exception as e:
        print(f"Error in lifecycle analysis: {e}")

    # Generate outputs
    try:
        analyzer.generate_report()
        analyzer.generate_enhanced_report()
    except Exception as e:
        print(f"Error generating report: {e}")
    
    try:
        analyzer.visualize_insights()
    except Exception as e:
        print(f"Error generating visualizations: {e}")

    # Save detailed results
    print("\n=== SAVING DETAILED RESULTS ===")
    
    # Save duplicates
    try:
        with open('duplicate_issues.json', 'w', encoding='utf-8') as f:
            json.dump(duplicates, f, indent=2, cls=NumpyEncoder)
        print("Saved duplicate analysis to duplicate_issues.json")
    except Exception as e:
        print(f"Error saving duplicates: {e}")
    
    # Save outdated issues (was stale_issues)
    if not outdated_issues.empty:
        try:
            # Convert to regular Python types before saving
            stale_export = outdated_issues.copy()
            stale_export['number'] = stale_export['number'].astype(int)
            stale_export['days_since_update'] = stale_export['days_since_update'].astype(int)
            stale_export['labels'] = stale_export['labels'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

            stale_export[['number', 'title', 'updated_at', 'days_since_update', 'labels']].to_csv(
                'stale_issues.csv', index=False
            )
            print("Saved stale issues to stale_issues.csv")
        except Exception as e:
            print(f"Error saving stale issues: {e}")
    
    # Save categorized issues
    try:
        categorized_df = analyzer.df[['number', 'title', 'state', 'categories', 'labels']].copy()
        categorized_df['categories'] = categorized_df['categories'].apply(lambda x: ', '.join(x) if x else '')
        categorized_df.to_csv('categorized_issues.csv', index=False)
        print("Saved categorized issues to categorized_issues.csv")
    except Exception as e:
        print(f"Error saving categorized issues: {e}")
    
    # Save enhanced results
    try:
        # Save topic analysis
        if hasattr(analyzer, 'topic_clusters') and analyzer.topic_clusters:
            with open('topic_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(analyzer.topic_clusters, f, indent=2, cls=NumpyEncoder)
            print("Saved topic analysis to topic_analysis.json")
    except Exception as e:
        print(f"Error saving topic analysis: {e}")
    
    # Save outdated issues with more details
    if not analyzer.outdated_issues.empty:
        try:
            outdated_export = analyzer.outdated_issues.copy()
            outdated_export['number'] = outdated_export['number'].astype(int)
            outdated_export['days_since_update'] = outdated_export['days_since_update'].astype(int)
            outdated_export['days_since_creation'] = outdated_export['days_since_creation'].astype(int)
            outdated_export['outdated_score'] = outdated_export['outdated_score'].astype(int)
            outdated_export['labels'] = outdated_export['labels'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
            
            outdated_export[['number', 'title', 'outdated_reason', 'outdated_score', 
                           'days_since_update', 'days_since_creation', 'labels']].to_csv(
                'outdated_issues_detailed.csv', index=False
            )
            print("Saved detailed outdated issues to outdated_issues_detailed.csv")
        except Exception as e:
            print(f"Error saving outdated issues: {e}")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()